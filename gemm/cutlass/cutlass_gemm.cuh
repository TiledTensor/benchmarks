#pragma once

#include "utils/cpp/cuda_utils.cuh"
#include "utils/cpp/cutlass/copy.cuh"
#include "utils/cpp/cutlass/traits_base.cuh"

#include <cute/tensor.hpp>

#include <iostream>

namespace benchmarks {
namespace cutlass_wrapper {

using namespace cute;

template <typename Element_,                             //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kTM, const int kTN, const int kTK,   //
          typename Base = TraitsBase<Element_>>
struct GemmTraits : public Base {
    using Element = Element_;

    static_assert(kTM % kWarpPerRow == 0,
                  "the M dimension of the CTA tile should be divisible by the "
                  "number of warps along that that dimension.");
    static_assert(kTN % kWarpPerCol == 0,
                  "the N dimension of the CTA tile should be divisible by the "
                  "number of warps along that that dimension.");

    // using ThreadLayout = Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>,
    // _1>,
    //                             Stride<Int<kWarpPerCol>, _1, _1>>;
    // using ValueLayout = Tile<Int<16 * kWarpPerRow>, Int<16 * kWarpPerCol>,
    // _16>;

    using ThreadLayout = Layout<Shape<_1, _1, _1>, Stride<_1, _1, _1>>;
    using ValueLayout = Tile<_16, _16, _16>;
    using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                              ThreadLayout, ValueLayout>;

    static constexpr int kThreads = size(TiledMma{});
    static_assert(kThreads == kWarpPerRow * kWarpPerCol * 32);

    static constexpr int kNumPerAccess = Base::kNumPerAccess;
    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    // using SmemLayoutAtom = decltype(composition(
    //     Swizzle<2, 3, 3>{}, Layout<Shape<_8, Int<4 * kNumPerAccess>>,
    //                                Stride<Int<4 * kNumPerAccess>, _1>>{}));
    // using SmemLayoutA =
    //     decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>,
    //     Int<kTK>>{}));
    // using SmemLayoutB =
    //     decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>,
    //     Int<kTK>>{}));
    //   using SmemLayoutC =
    //     decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>,
    //     Int<kTN>>{}));

    using SmemLayoutA = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kTK>, _1>>;
    using SmemLayoutB = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kTK>, _1>>;
    using SmemLayoutC = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kTN>, _1>>;

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInstG2S =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInstG2S = Copy_Atom<DefaultCopy, Element>;
#endif

    using TiledCopyG2S = decltype(make_tiled_copy(
        CopyInstG2S{},
        Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
               Stride<Int<kThreadsPerCol>, _1>>{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    // copy from shared memory to global memory dose not have cp.async support,
    // another `TiledCopy` has to be declared and uses `DefaultCopy` as the
    // `CopyAtom`.
    using TiledCopyS2G = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
               Stride<Int<kThreadsPerCol>, _1>>{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    using StoreC_R2S = R2SCopy2D<Element, TiledMma, SmemLayoutC>;
};

template <typename Element, const int kM, const int kN, const int kK,
          const int kTM, const int kTN, const int kTK, typename KeTraits>
__global__ void gemm_kernel(const Element* dA, const Element* dB, Element* dC) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // if (threadIdx.x == 0) {
    //     const __half* A = reinterpret_cast<const __half*>(dA);
    //     for (int i = 0; i < 16 * 16; i++) {
    //         printf("%.3f, ", __half2float(A[i]));

    //         if (i && (i + 1) % 8 == 0) printf("\n");
    //     }
    // }

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;
    Element* sC_ptr = shm;

    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopyG2S tiled_copy;

    auto rA = make_s2rA(sA_ptr, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        copy_tile_g2s(gA_ptr, sA_ptr,
                      Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>{},
                      typename KeTraits::SmemLayoutA{}, tiled_copy);
        copy_tile_g2s(gB_ptr, sB_ptr,
                      Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>{},
                      typename KeTraits::SmemLayoutB{}, tiled_copy);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            // compute using tcu's wmma instruction
            gemm(mma, rA[i], rB[i], acc);
        }
        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    typename KeTraits::StoreC_R2S sC;  // declare register to shared store plan
    sC.copy(acc, shm);                 // store register tile to shared memory
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Results in kernel SharedC:\n");
        const __half* A = reinterpret_cast<const __half*>(sC_ptr);
        for (int i = 0; i < 16 * 16; i++) {
            printf("%.3f, ", __half2float(A[i]));

            if (i && (i + 1) % 8 == 0) printf("\n");
        }
    }

    // store shared memory tile to global memory
    copy_tile_s2g(sC_ptr, gC_ptr, typename KeTraits::SmemLayoutC{},
                  Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kN>, _1>>{},
                  typename KeTraits::TiledCopyS2G{});

    // if (threadIdx.x == 0) {
    //     printf("Results in kernel GlobalC:\n");
    //     const __half* A = reinterpret_cast<const __half*>(gC_ptr);
    //     for (int i = 0; i < 16 * 16; i++) {
    //         printf("%.3f, ", __half2float(A[i]));

    //         if (i && (i + 1) % 8 == 0) printf("\n");
    //     }
    // }
}
}  // namespace cutlass_wrapper
}  // namespace benchmarks

template <typename Element,                              //
          const int kWarpPerRow, const int kWarpPerCol,  //
          const int kM, const int kN, const int kK,      //
          const int kTM, const int kTN, const int kTK>
void cute_gemm(const Element* dA, const Element* dB, Element* dC) {
    using namespace benchmarks::cutlass_wrapper;

    using KeTraits =
        GemmTraits<Element, kWarpPerRow, kWarpPerCol, kTM, kTN, kTK>;

    std::cout << "kThreads: " << KeTraits::kThreads << std::endl
              << "kNumPerAccess:" << KeTraits::kNumPerAccess << std::endl
              << "threads layouts: " << KeTraits::kThreadsPerRow << ", "
              << KeTraits::kThreadsPerCol << std::endl;

    static constexpr int smem_size =
        std::max(kTK * (kTN + kTM), kTM * kTN) * sizeof(Element);

    auto kernel = &gemm_kernel<Element, kM, kN, kK, kTM, kTN, kTK, KeTraits>;

    // maximal statically allocated smem per block
    const int kMaxSmemPerBlock = 48 * 1024;
    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    const int block_m = (kM + kTM - 1) / kTM;
    const int block_n = (kN + kTN - 1) / kTN;

    const int kThreads = KeTraits::kThreads;

    dim3 gridDim(block_m, block_n);
    dim3 blockDim(kThreads, 1, 1);

    kernel<<<gridDim, blockDim, smem_size>>>(dA, dB, dC);
    // cudaDeviceSynchronize();
}
