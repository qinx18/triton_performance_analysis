import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Apply condition and sum
    condition = a_vals > 0.0
    masked_vals = tl.where(condition, a_vals, 0.0)
    block_sum = tl.sum(masked_vals)
    
    # Store block sum (will be reduced later)
    tl.store(result_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary result tensor for block sums
    block_results = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3111_kernel[(grid_size,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Sum all block results to get final sum
    final_sum = torch.sum(block_results)
    
    return final_sum