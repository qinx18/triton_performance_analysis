import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    condition_mask = vals > 0.0
    conditional_vals = tl.where(condition_mask, vals, 0.0)
    block_sum = tl.sum(conditional_vals)
    
    tl.atomic_add(result_ptr, block_sum)

def s3111_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s3111_kernel[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result[0]