import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(q_ptr, n_iterations, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid >= 1:
        return
    
    q = 1.0
    for i in range(n_iterations):
        q = q * 0.99
    
    tl.store(q_ptr, q)

def s317_triton(n):
    n_iterations = n // 2
    
    q_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    grid = (1,)
    BLOCK_SIZE = 128
    
    s317_kernel[grid](
        q_tensor,
        n_iterations,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return q_tensor.item()