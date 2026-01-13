import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n_elements
    
    # Load a[0] from the read-only copy
    a_0_val = tl.load(a_copy_ptr)
    
    # Store a[0] value to all positions in this block
    tl.store(a_ptr + current_offsets, a_0_val, mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to avoid WAR race conditions
    a_copy = a.clone()
    
    # Launch kernel with appropriate grid size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a, a_copy, n, BLOCK_SIZE
    )