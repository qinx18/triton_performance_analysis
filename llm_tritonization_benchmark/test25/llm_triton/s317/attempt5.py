import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(q_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    if block_id == 0:
        q = 1.0
        factor = 0.99
        iterations = n // 2
        
        for i in range(iterations):
            q *= factor
        
        tl.store(q_ptr, q)

def s317_triton(n):
    q = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s317_kernel[grid](q, n, BLOCK_SIZE)
    
    return q.item()