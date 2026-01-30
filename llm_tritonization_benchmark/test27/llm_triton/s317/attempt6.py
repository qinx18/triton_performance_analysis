import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        q = 1.0
        for i in range(n // 2):
            q = q * 0.99
        tl.store(output_ptr, q)

def s317_triton(n):
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    grid = (1,)
    s317_kernel[grid](output, n, BLOCK_SIZE=128)
    
    return output.item()