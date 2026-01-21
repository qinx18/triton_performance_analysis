import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    mask = current_offsets < n
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    positive_mask = vals > 0.0
    filtered_vals = tl.where(positive_mask, vals, 0.0)
    
    block_sum = tl.sum(filtered_vals, axis=0)
    tl.store(out_ptr + pid, block_sum)

def s3111_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    partial_sums = torch.zeros(grid[0], dtype=a.dtype, device=a.device)
    
    s3111_kernel[grid](a, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return partial_sums.sum().item()