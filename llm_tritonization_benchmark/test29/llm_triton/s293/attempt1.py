import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load the first element (constant value to broadcast)
    first_val = tl.load(a_copy_ptr)
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Store the first element value to all positions
        tl.store(a_ptr + current_offsets, first_val, mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to avoid WAR race conditions
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a, a_copy, n, BLOCK_SIZE=BLOCK_SIZE
    )