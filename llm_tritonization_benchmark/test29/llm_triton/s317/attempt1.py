import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    q = 1.0
    factor = 0.99
    
    loop_count = n // 2
    
    for i in range(loop_count):
        q *= factor
    
    tl.store(result_ptr, q)

def s317_triton(n):
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s317_kernel[grid](
        result,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()