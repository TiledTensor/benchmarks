#pragma once

namespace benchmarks {
namespace cutlass_wrapper {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_SM80_ENABLED
#endif

template <typename Element>
struct TraitsBase {
    // the maximal width of vectorized access.
    static constexpr int kAccessInBits = 128;
    static constexpr int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;
};

}  // namespace cutlass_wrapper
}  // namespace benchmarks
