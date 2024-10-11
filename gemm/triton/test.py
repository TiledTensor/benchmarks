import torch

import gemm

def bench():
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device = 'cuda', dtype=torch.float16)
    b = torch.randn(1024, 1024, device = 'cuda', dtype=torch.float16)
    
    triton_c = gemm.gemm(a, b)
    torch_c = torch.mm(a, b)
    
    print(torch.allclose(triton_c, torch_c, atol = 1e-3))
    

if __name__ == '__main__':
    bench()
