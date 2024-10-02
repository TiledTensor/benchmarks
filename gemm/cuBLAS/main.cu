#include "cublas_gemm.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <const int kM, const int kN, const int kK>
void run_test() {
    using DType = __half;

    thrust::host_vector<DType> h_a(kM * kK);
    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<DType>(rand_float());

    thrust::host_vector<DType> h_b(kK * kN);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<DType>(rand_float());

    thrust::host_vector<DType> h_c(kM * kN);
    thrust::fill(h_c.begin(), h_c.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<AccType> d_c = h_c;

    const DType* A = thrust::raw_pointer_cast(d_a.data());
    const DType* B = thrust::raw_pointer_cast(d_b.data());
    DType* C = thrust::raw_pointer_cast(d_c.data());

    std::cout << "elapsed time: " << cublas_hgemm<kM, kN, kK>(A, B, C);
}

int main(int argc, char* argv[]) {
    run_test<128, 128, 128>();

    return 0;
}
