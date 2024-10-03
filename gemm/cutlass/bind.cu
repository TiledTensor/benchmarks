#include "cutlass_gemm.cuh"

extern "C" int kernel_entry(const __half* dA, const __half* dB, __half* dC) {
    using DType = cutlass::half_t;

    auto* A = reinterpret_cast<const DType*>(dA);
    auto* B = reinterpret_cast<const DType*>(dB);
    auto* C = reinterpret_cast<DType*>(dC);

    cute_gemm<DType, 1, 1, 16, 16, 16, 16, 16, 16>(A, B, C);
    return 0;
}
