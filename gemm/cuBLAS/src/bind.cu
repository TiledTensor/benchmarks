#include "cublas_gemm.cuh"

#include <torch/script.h>

void gemm_op(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
             int64_t m, int64_t n, int64_t k) {
    using DType = __half;

    auto* dA = reinterpret_cast<const DType*>(A.data_ptr());
    auto* dB = reinterpret_cast<const DType*>(B.data_ptr());
    auto* dC = reinterpret_cast<DType*>(C.data_ptr());

    cublas_hgemm(dA, dB, dC, m, n, k);
}

TORCH_LIBRARY(cublas_gemm, t) { t.def("gemm", &gemm_op); };
