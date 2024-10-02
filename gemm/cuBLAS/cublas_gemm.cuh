#pragma once

#include "util/cuda_timer.hpp"

#include <cublas_v2.h>

/* In this implementation, A and C is laid out in row-major, B, is laid out in
  column-major.

  C[m, n] = A[m, k] @ B[k, n]
*/
template <const int kM, const int kN, const int kK>
float cublas_hgemm(const __half* A, const __half* B, const __half* C,
                   int warp_up = 5, int iters = 20) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alf = static_cast<__half>(1.);
    __half bet = static_cast<__half>(0.);

    // acc   = A @ B
    // acc^T = B^T @ A^T
    // [n, m] = [n, k] @ [k, m]
    for (int i = 0; i < warp_up; i++) {
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN, kM, kK,
                    &alf, B, kK, A, kK, &bet, acc, kN);
        cudaDeviceSynchronize();
    }

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; i++) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, kM, kN, kK, &alf, A, kK,
                    B, kN, &bet, C, kM);
    }
    cudaDeviceSynchronize();
    float time = timer.stop();

    cublasDestroy(handle);

    return time / iters;
}
