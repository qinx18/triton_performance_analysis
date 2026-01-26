import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(output_ptr, n_half):
    q = 1.0
    for i in range(n_half):
        q *= 0.99
    
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(output_ptr, q)

def s317_triton(n):
    half_len = n // 2
    
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    s317_kernel[(1,)](result, half_len)
    
    return result[0].item()