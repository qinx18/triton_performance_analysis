import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Apply condition: only sum values > 0
    condition = a_vals > 0.0
    filtered_vals = tl.where(condition, a_vals, 0.0)
    
    block_sum = tl.sum(filtered_vals)
    
    # Store the block sum
    tl.store(result_ptr + pid, block_sum)

def s3111_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid[0], dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3111_kernel[grid](
        a, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum the partial results
    total_sum = torch.sum(partial_sums)
    
    return total_sum