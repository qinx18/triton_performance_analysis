import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    block_max = tl.max(abs_vals, axis=0)
    
    tl.store(result_ptr + pid, block_max)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # Allocate space for block results
    block_results = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (grid_size,)
    s3113_kernel[grid](a, block_results, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Find maximum among all block results
    max_val = torch.max(block_results)
    
    return max_val