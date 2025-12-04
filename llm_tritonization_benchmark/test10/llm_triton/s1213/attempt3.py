import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid elements (1 <= i < n_elements-1)
    mask = (offsets >= 1) & (offsets < n_elements - 1)
    
    # Load a[i+1] values first (before a is modified)
    a_next_offsets = offsets + 1
    a_next_mask = (offsets >= 1) & (offsets < n_elements - 1) & (a_next_offsets < n_elements)
    a_next = tl.load(a_ptr + a_next_offsets, mask=a_next_mask)
    
    # Load other required elements
    b_prev_offsets = offsets - 1
    b_prev_mask = (offsets >= 1) & (offsets < n_elements - 1) & (b_prev_offsets >= 0)
    b_prev = tl.load(b_ptr + b_prev_offsets, mask=b_prev_mask)
    
    c_curr = tl.load(c_ptr + offsets, mask=mask)
    d_curr = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute new values
    new_a = b_prev + c_curr
    new_b = a_next * d_curr
    
    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1213_kernel[grid](
        a, b, c, d, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b