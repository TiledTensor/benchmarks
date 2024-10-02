import torch

torch.ops.load_library("src/build/libcublas_gemm.so")


def gemm(a, b, c, m, n, k):
    torch.ops.cublas_gemm.gemm(a, b, c, m, n, k)


if __name__ == '__main__':
    M = 128
    N = 128
    K = 128

    device = torch.device("cuda")
    dtype = torch.float16

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.zeros(M, N, device=device, dtype=dtype)

    a_data = a.flatten().half()
    b_data = b.flatten().half()
    c_data = c.flatten().half()

    gemm(a_data, b_data, c_data, M, N, K)
    ref_c = a @ b.t()

    epsilon = 5e-2
    avg_diff = (torch.sum(torch.abs(ref_c - c)) / (M * N)).item()
    if (avg_diff > epsilon):
        raise ValueError("Failed unittest.")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for i in range(iters):
        gemm(a_data, b_data, c_data, M, N, K)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters
    print("Average time: {:.3} ms".format(time))
