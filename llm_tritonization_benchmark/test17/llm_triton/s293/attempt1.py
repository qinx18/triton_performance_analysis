import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get thread block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n_elements
    
    # Load the constant value from a_copy[0]
    const_val = tl.load(a_copy_ptr)
    
    # Store the constant value to all positions
    tl.store(a_ptr + current_offsets, const_val, mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s293_kernel[grid](
        a, a_copy, n, BLOCK_SIZE
    )
    
    return a