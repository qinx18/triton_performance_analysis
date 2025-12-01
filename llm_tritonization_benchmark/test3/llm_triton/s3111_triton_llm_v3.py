import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Apply conditional sum: if a[i] > 0, include in sum
    condition = a_vals > 0.0
    conditional_vals = tl.where(condition, a_vals, 0.0)
    
    # Sum the values in this block
    block_sum = tl.sum(conditional_vals)
    
    # Store the block sum
    tl.store(output_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros((grid_size,), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s3111_kernel[(grid_size,)](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Sum the partial results on CPU/GPU
    total_sum = torch.sum(partial_sums)
    
    return total_sum