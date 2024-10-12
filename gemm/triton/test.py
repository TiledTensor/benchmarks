import torch

import gemm


def run_unittest(
    M: int,
    N: int,
    K: int,
    debug_print=False,
    epsilon: float = 5e-2
):  
    torch.manual_seed(0)
    a = torch.randn(M, K, device = 'cuda', dtype = torch.float16)
    b = torch.randn(K, N, device = 'cuda', dtype = torch.float16)
    
    triton_c = gemm.gemm(a, b)
    torch_c = torch.mm(a, b)
    
    if debug_print:
        print("Result:")
        print(triton_c)
        
        print("\nReference:")
        print(torch_c)
    
    avg_diff = (torch.sum(torch.abs(triton_c.half() - torch_c) / (M * N))).item()
    
    if avg_diff > epsilon:
        return False
    else:
        return True

def bench(
    M: int,
    N: int,
    K: int
):
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device = 'cuda', dtype=torch.float16)
    b = torch.randn(1024, 1024, device = 'cuda', dtype=torch.float16)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iters = 50
    start_event.record()
    for _ in range(iters):
        triton_c = gemm.gemm(a, b)
    end_event.record()
    torch.cuda.synchronize()

    time = start_event.elapsed_time(end_event) / iters

    return time
    
    
if __name__ == '__main__':    
    M = 4096
    N = 4096
    K = 2048
    
    if run_unittest(M, N, K):
        print("Unittest passed")
    else:
        print("Unittest failed")
        
    time = bench(M, N, K)
    
    print("Elapsed time: {:.4f} ms".format(time))
