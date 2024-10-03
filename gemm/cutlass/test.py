import torch
from torch import Tensor
from typing import Tuple

from gemm import gemm_func


def run_unittest(a: Tensor,
                 b: Tensor,
                 c: Tensor,
                 M: int,
                 N: int,
                 K: int,
                 kTM: int,
                 kTN: int,
                 kTK: int,
                 warp_layout: Tuple,
                 debug_print=True,
                 epsilon: float = 5e-2):
    gemm_func(a, b, c, M, N, K, kTM, kTN, kTK, *warp_layout)

    ref_c = a @ b.t()

    if debug_print:
        print("Result:")
        print(c)

        # print("\nReference:")
        # print(ref_c)

    avg_diff = (torch.sum(torch.abs(ref_c - c)) / (M * N)).item()
    print("Average difference: {:.4f}".format(avg_diff))
    if avg_diff > epsilon:
        return False
    else:
        return True


def run_test(
    M: int,
    N: int,
    K: int,
    kTM: int,
    kTN: int,
    kTK: int,
    warp_layout: Tuple,
):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(1234)

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    c = torch.zeros(M, N, device=device, dtype=dtype)

    print(run_unittest(a, b, c, M, N, K, kTM, kTN, kTK, warp_layout))

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)

    # iters = 50
    # start_event.record()
    # for _ in range(iters):
    #     gemm_func(a, b, c, M, N, K, kTM, kTN, kTK, *warp_layout)
    # end_event.record()
    # torch.cuda.synchronize()

    # time = start_event.elapsed_time(end_event) / iters

    time = 0.
    return time


if __name__ == "__main__":
    kM = 32
    kN = 32
    kK = 32

    kTM = 32
    kTN = 32
    kTK = 32

    time = run_test(kM, kN, kK, kTM, kTN, kTK, (1, 1))

    print("Elapsed time: {:.4f}".format(time))
