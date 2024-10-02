#pragma once

#include <cublas_v2.h>

/* In this implementation, A and C is laid out in row-major, B, is laid out in
  column-major: C[m, n] = A[m, k] @ B[k, n]
*/

void cublas_hgemm(const __half* A, const __half* B, __half* C, int64_t kM,
                  int64_t kN, int64_t kK) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alf = static_cast<__half>(1.);
    __half bet = static_cast<__half>(0.);

    // C   = A @ B, but in cuBLAS, matrix is by default laid out in column-major
    // C^T = B^T @ A^T
    // [n, m] = [n, k] @ [k, m]
    cublasHgemm(handle, CUBLAS_OP_T /* transb*/, CUBLAS_OP_N, kN, kM, kK, &alf,
                B, kK, A, kK, &bet, C, kN);
    cudaDeviceSynchronize();
}
