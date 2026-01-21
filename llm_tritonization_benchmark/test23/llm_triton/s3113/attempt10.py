import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n
    
    # Load values and compute absolute values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find max within this block
    block_max = tl.max(abs_vals)
    
    # Store the block max
    tl.store(result_ptr + pid, block_max)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # First pass: compute block maxima
    block_results = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    grid = (grid_size,)
    s3113_kernel[grid](a, block_results, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second pass: find global maximum, but ensure we handle the first element case
    global_max = torch.max(block_results).item()
    
    # Handle case where all values might be smaller than abs(a[0])
    first_abs = torch.abs(a[0]).item()
    return max(global_max, first_abs)