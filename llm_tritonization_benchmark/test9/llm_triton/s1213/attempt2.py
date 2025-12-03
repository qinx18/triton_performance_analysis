import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements 1 to n_elements-2 (inclusive)
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements - 1
        
        # Load b[i-1] + c[i] for a[i] = b[i-1] + c[i]
        b_prev_offsets = current_offsets - 1
        b_prev = tl.load(b_ptr + b_prev_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        a_new = b_prev + c_vals
        
        # Store a[i] = b[i-1] + c[i]
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Load a[i+1] (using original values, not updated ones)
        a_next_offsets = current_offsets + 1
        a_next = tl.load(a_ptr + a_next_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        b_new = a_next * d_vals
        
        # Store b[i] = a[i+1] * d[i]
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s1213_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )