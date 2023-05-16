/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include <memory>
#include <vector>
#include <iostream>
#include <ctime>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/softmax_op_functor.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("MHAOp")
    .Attr("T: {float32, bfloat16}")
    .Input("weight0_0: T")
    .Input("weight0_1: T")
    .Input("weight0_2: T")
    .Input("weight0_3: int32")
    .Output("fused_output: T");

using namespace tensorflow;
using namespace functor;

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {

// Returns the pair of dimensions along which to perform Tensor contraction to
// emulate matrix multiplication.
// For matrix multiplication of 2D Tensors X and Y, X is contracted along
// second dimension and Y is contracted along the first dimension (if neither X
// nor Y is adjointed). The dimension to contract along is switched when any
// operand is adjointed.
// See http://en.wikipedia.org/wiki/Tensor_contraction
inline Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x,
                                                           bool adj_y) {
  return Eigen::IndexPair<Eigen::DenseIndex>(adj_x ? 0 : 1, adj_y ? 1 : 0);
}

// Parallel batch matmul kernel based on the multi-threaded tensor contraction
// in Eigen.
// The Eigen contraction kernel used here is very large and slow to compile,
// so we partially specialize ParallelMatMulKernel for real types to avoid all
// but one of the instantiations.
template <typename T>
struct HeadfusedMHAKernel_1 {
  static void Run(const OpKernelContext* context, const Tensor& in_x,
                  const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                  bool trans_y, const MatMulBCast& bcast, Tensor* out,
                  int batch_size) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x || trans_x, adj_y || trans_y);
    if (batch_size == 1 && !should_bcast) {
      auto Tx = in_x.flat_inner_dims<T, 2>();
      auto Ty = in_y.flat_inner_dims<T, 2>();
      auto Tz = out->flat_inner_dims<T, 2>();
      Tz.device(d) = Tx.contract(Ty, contract_pairs);
    } else {
      auto Tx = in_x.tensor<T, 3>();
      auto Ty = in_y.tensor<T, 3>();
      auto Tz = out->tensor<T, 3>();
      const auto& x_batch_indices = bcast.x_batch_indices();
      const auto& y_batch_indices = bcast.y_batch_indices();
      // TODO(rmlarsen): Consider launching these contractions asynchronously.
      for (int64_t i = 0; i < batch_size; ++i) {
        const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
        const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
        auto x = Tx.template chip<0>(x_batch_index);
        auto y = Ty.template chip<0>(y_batch_index);
        auto z = Tz.template chip<0>(i);

        z.device(d) = x.contract(y, contract_pairs);
      }
    }
  }
};

// Sequential batch matmul kernel that calls the regular Eigen matmul.
// We prefer this over the tensor contraction because it performs
// better on vector-matrix and matrix-vector products.
template <typename T>
struct HeadfusedMHAKernel {
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using Matrix_1 =
      Eigen::Matrix<int32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap_1 = Eigen::Map<const Matrix_1>;
  using MatrixMap = Eigen::Map<Matrix>;

  static ConstMatrixMap ConstTensorSliceToEigenMatrix(const Tensor& t,
                                                      int slice) {
    return ConstMatrixMap(
        t.flat<T>().data() + slice * t.dim_size(1) * t.dim_size(2),
        t.dim_size(1), t.dim_size(2));
  }

  static ConstMatrixMap_1 ConstTensorSliceToEigenMatrix_1(const Tensor& t,
                                                      int slice) {
    return ConstMatrixMap_1(
        t.flat<int32>().data() + slice * t.dim_size(1) * t.dim_size(2),
        t.dim_size(1), t.dim_size(2));
  }

  static MatrixMap TensorSliceToEigenMatrix(Tensor* t, int slice) {
    return MatrixMap(
        t->flat<T>().data() + slice * t->dim_size(1) * t->dim_size(2),
        t->dim_size(1), t->dim_size(2));
  }

  static void Run(const OpKernelContext* context, const Tensor& in_x, const Tensor& in_y, const Tensor& in_z, const Tensor& in_m, Tensor* out, int start, int limit) {
    for (int64_t i = start; i < limit; ++i) {
      auto x = ConstTensorSliceToEigenMatrix(in_x, i);
      auto y = ConstTensorSliceToEigenMatrix(in_y, i);
      auto y_1 = ConstTensorSliceToEigenMatrix(in_z, i);
      auto y_2 = ConstTensorSliceToEigenMatrix_1(in_m, i);
      auto ll = y_2.template cast<float>();
      auto k =  (1.0f - ll.array()).matrix() * (-32767.0f);
      auto l = (x * y.transpose()) / (5.656854152679443f);
      auto z = l + k;
      auto shifted_logits = z.colwise() - z.rowwise().maxCoeff();
      auto softmax = shifted_logits.array().exp();
      auto softmax_1 = softmax.colwise() * softmax.rowwise().sum().inverse();
      auto m = TensorSliceToEigenMatrix(out, i);
      m.noalias() = softmax_1.matrix() * y_1;
      } 
    }
};
}

template <typename Device, typename T>
void compute_batchmatmul(OpKernelContext* ctx, const Tensor& in1,
                     const Tensor& in2, bool adj_x_, bool adj_y_, Tensor& out) {
    MatMulBCast bcast(in1.shape().dim_sizes(), in2.shape().dim_sizes());
    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();
    auto d0 = in1.dim_size(in1.dims() - 2);
    auto d1 = in1.dim_size(in1.dims() - 1);
    Tensor in0_reshaped;
    OP_REQUIRES(
        ctx,
        in0_reshaped.CopyFrom(in1, TensorShape({bcast.x_batch_size(), d0, d1})),
        errors::Internal("Failed to reshape In[0] from ",
                         in1.shape().DebugString()));
    auto d2 = in2.dim_size(in2.dims() - 2);
    auto d3 = in2.dim_size(in2.dims() - 1);
    Tensor in1_reshaped;
    OP_REQUIRES(
        ctx,
        in1_reshaped.CopyFrom(in2, TensorShape({bcast.y_batch_size(), d2, d3})),
        errors::Internal("Failed to reshape In[1] from ",
                         in2.shape().DebugString()));
    if (adj_x_) std::swap(d0, d1);
    if (adj_y_) std::swap(d2, d3);
    OP_REQUIRES(
        ctx, d1 == d2,
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", in1.shape().DebugString(),
            ", In[1]: ", in2.shape().DebugString()));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
    //OP_REQUIRES_OK(ctx, out_shape.AddDimWithStatus(d0));
    //OP_REQUIRES_OK(ctx, out_shape.AddDimWithStatus(d3));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, out_shape, &out));
    /*Tensor out_reshaped;
    OP_REQUIRES(ctx,
                out_reshaped.CopyFrom(*out, TensorShape({batch_size, d0, d3})),
                errors::Internal("Failed to reshape output from ",
                                 out->shape().DebugString()));*/
    //clock_t start = clock();
    LaunchBatchMatMul<Device, T>::Launch(
          ctx, in0_reshaped, in1_reshaped, adj_x_, adj_y_, false,
          false, bcast, &out);
    //clock_t end  = clock();
    //double programTimes = ((double) end - start) / CLOCKS_PER_SEC;
    //std::cout<< programTimes << std::endl;
}

template <typename Device, typename T>
class MHAOp : public OpKernel {
 public:
  explicit MHAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~MHAOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& Q_tensor = ctx->input(0);
    const Tensor& K_tensor = ctx->input(1);
    const Tensor& w1_tensor = ctx->input(2);
    const Tensor& w2_tensor = ctx->input(3);
    Tensor* out;
    Tensor w2_reshaped;
    OP_REQUIRES(
        ctx,
        w2_reshaped.CopyFrom(w2_tensor, TensorShape({w2_tensor.dim_size(0), 1, w2_tensor.dim_size(1)})),
        errors::Internal("Failed to reshape In[1] from ",
                         w2_tensor.shape().DebugString()));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, Q_tensor.shape(), &out));
    const int64_t batch_size = Q_tensor.dim_size(0);
    const int64_t cost_per_unit =
        Q_tensor.dim_size(1) * Q_tensor.dim_size(2) * K_tensor.dim_size(1) * 2 + K_tensor.dim_size(1) * 10;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit,
            [ctx, &Q_tensor, &K_tensor, &w1_tensor, &w2_reshaped, out](
                int start, int limit) {
              HeadfusedMHAKernel<T>::Run(ctx, Q_tensor, K_tensor, w1_tensor, w2_reshaped, out,
                                                  start, limit);
            });
    /*Tensor out_1;
    Tensor out_2; 
    Tensor out_3;
    Tensor* softmax_out;
    compute_batchmatmul<Device, T>(ctx, Q_tensor, w1_tensor, false, false, out_1);
    //std::cout << *out_1 << std::endl;
    compute_batchmatmul<Device, T>(ctx, K_tensor, w2_tensor, false, false, out_2);
    compute_batchmatmul<Device, T>(ctx, out_1, out_2, false, true, out_3);
    const Tensor& out_4 = static_cast<const Tensor&>(out_3);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_4.shape(), &softmax_out));
    SoftmaxEigenImpl<Device, T>::Compute(ctx->eigen_device<Device>(), out_4.flat_inner_dims<T>(),
                                        softmax_out->flat_inner_dims<T>(), false);*/
  }
};

REGISTER_KERNEL_BUILDER(Name("MHAOp").Device(DEVICE_CPU).TypeConstraint<float>("T"), MHAOp<CPUDevice, float>);



