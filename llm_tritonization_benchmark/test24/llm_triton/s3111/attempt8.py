import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    positive_mask = vals > 0.0
    filtered_vals = tl.where(mask & positive_mask, vals, 0.0)
    block_sum = tl.sum(filtered_vals)
    
    tl.atomic_add(result_ptr, block_sum)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s3111_kernel[grid](a, result, N, BLOCK_SIZE)
    
    return result.item()