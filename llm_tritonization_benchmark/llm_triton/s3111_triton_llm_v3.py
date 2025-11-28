import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data from memory
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Conditional sum: only add positive values
    condition = a_vals > 0.0
    conditional_vals = tl.where(condition, a_vals, 0.0)
    
    # Sum within this block
    block_sum = tl.sum(conditional_vals)
    
    # Store the block sum (will be reduced later)
    tl.store(output_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3111_kernel[(grid_size,)](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Reduce partial sums to get final result
    final_sum = torch.sum(partial_sums)
    
    return final_sum