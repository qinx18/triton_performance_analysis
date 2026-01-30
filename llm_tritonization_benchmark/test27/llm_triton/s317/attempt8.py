import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(result_ptr, num_iters):
    q = 1.0
    for i in range(num_iters):
        q *= 0.99
    tl.store(result_ptr, q)

def s317_triton(n):
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    num_iters = n // 2
    s317_kernel[(1,)](result, num_iters)
    
    return result.item()