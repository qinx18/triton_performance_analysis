import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load a[0] once
    a0_val = tl.load(a_copy_ptr)
    
    # Get block offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n_elements
    
    # Store a[0] to all elements in this block
    tl.store(a_ptr + current_offsets, a0_val, mask=mask)

def s293_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s293_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE
    )