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
    
    if n == 0:
        return torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    if grid_size == 1:
        block_results = torch.zeros(1, dtype=a.dtype, device=a.device)
        grid = (1,)
        s3113_kernel[grid](a, block_results, n, BLOCK_SIZE=BLOCK_SIZE)
        return block_results[0]
    else:
        block_results = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
        grid = (grid_size,)
        s3113_kernel[grid](a, block_results, n, BLOCK_SIZE=BLOCK_SIZE)
        
        while grid_size > 1:
            new_grid_size = triton.cdiv(grid_size, BLOCK_SIZE)
            new_results = torch.zeros(new_grid_size, dtype=a.dtype, device=a.device)
            grid = (new_grid_size,)
            s3113_kernel[grid](block_results, new_results, grid_size, BLOCK_SIZE=BLOCK_SIZE)
            block_results = new_results
            grid_size = new_grid_size
        
        return block_results[0]