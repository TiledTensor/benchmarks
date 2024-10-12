import torch

import gemm


def run_unittest(
    a: torch.Tensor,
    b: torch.Tensor,
    M: int,
    N: int,
    K: int,
    debug_print=False,
    epsilon: float = 5e-2
):
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

def bench():
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device = 'cuda', dtype=torch.float16)
    b = torch.randn(1024, 1024, device = 'cuda', dtype=torch.float16)
    
    triton_c = gemm.gemm(a, b)
    torch_c = torch.mm(a, b)
    
    

if __name__ == '__main__':
    torch.manual_seed(0)
    
    M = 1024
    N = 1024
    K = 1024
    
    a = torch.randn(M, K, device = 'cuda', dtype = torch.float16)
    b = torch.randn(K, N, device = 'cuda', dtype = torch.float16)
    
    if run_unittest(a, b, M, N, K):
        print("Unittest passed")
    else:
        print("Unittest failed")
