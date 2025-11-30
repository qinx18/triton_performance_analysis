import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Conditional sum: only add if a[i] > 0
    condition = a_vals > 0.0
    conditional_vals = tl.where(condition, a_vals, 0.0)
    
    # Sum within this block
    block_sum = tl.sum(conditional_vals)
    
    # Store block sum (will be reduced later)
    tl.store(output_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s3111_kernel[(grid_size,)](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    final_sum = torch.sum(partial_sums)
    
    return final_sum.item()