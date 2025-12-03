import torch
import triton
import triton.language as tl

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, s1, s2, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load previous a value (i-1)
        prev_offsets = current_offsets - 1
        prev_mask = (prev_offsets >= 0) & (current_offsets < n_elements)
        a_prev = tl.load(a_ptr + prev_offsets, mask=prev_mask)
        
        # Load b, c, d values at current position
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute new a values
        new_a = a_prev + s1 + s2 + b_vals + c_vals + d_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, new_a, mask=mask)

def s242_triton(a, b, c, d, s1, s2):
    n_elements = a.shape[0]
    
    # Convert scalar parameters to float if they're tensors
    if hasattr(s1, 'item'):
        s1 = s1.item()
    if hasattr(s2, 'item'):
        s2 = s2.item()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s242_kernel[grid](
        a, b, c, d,
        n_elements, s1, s2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a