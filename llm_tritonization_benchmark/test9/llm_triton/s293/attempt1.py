import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load a[0] from the read-only copy
    a_first = tl.load(a_copy_ptr)
    
    # Get block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < n_elements
    
    # Store a[0] to all positions a[i] in this block
    tl.store(a_ptr + indices, a_first, mask=mask)

def s293_triton(a):
    n_elements = a.numel()
    
    # Create read-only copy to handle WAR race condition
    a_copy = a.clone()
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s293_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE
    )