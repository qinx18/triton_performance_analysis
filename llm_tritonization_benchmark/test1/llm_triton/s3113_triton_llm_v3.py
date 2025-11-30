import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute values
    abs_vals = tl.abs(a_vals)
    
    # Find maximum in this block
    block_max = tl.max(abs_vals, axis=0)
    
    # Store block maximum
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block maximums
    block_maxs = torch.zeros(grid_size, device=a.device, dtype=a.dtype)
    
    # Launch kernel to find maximum in each block
    s3113_kernel[(grid_size,)](
        a, block_maxs, n_elements, BLOCK_SIZE
    )
    
    # Find global maximum from block maximums
    max_val = torch.max(block_maxs)
    
    return max_val