import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n
    
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(a_vals)
    
    tl.atomic_add(result_ptr, block_sum)

def s311_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s311_kernel[grid](
        a, result, n, BLOCK_SIZE
    )
    
    return result.item()