import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n_elements
    
    # Load a[0] from the copy
    a_0_val = tl.load(a_copy_ptr)
    
    # Store a[0] value to all positions a[i]
    tl.store(a_ptr + current_offsets, a_0_val, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s293_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )