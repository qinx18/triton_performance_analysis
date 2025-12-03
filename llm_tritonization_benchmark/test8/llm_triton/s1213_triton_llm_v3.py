import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = (current_offsets >= 1) & (current_offsets < n_elements - 1)
        
        # Load b[i-1] (shifted by -1)
        b_prev_offsets = current_offsets - 1
        b_prev_mask = (b_prev_offsets >= 0) & (b_prev_offsets < n_elements)
        b_prev = tl.load(b_ptr + b_prev_offsets, mask=b_prev_mask & mask, other=0.0)
        
        # Load c[i]
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load a[i+1] (shifted by +1) - this creates a WAR dependency
        a_next_offsets = current_offsets + 1
        a_next_mask = (a_next_offsets >= 0) & (a_next_offsets < n_elements)
        a_next = tl.load(a_ptr + a_next_offsets, mask=a_next_mask & mask, other=0.0)
        
        # Load d[i]
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = b[i-1] + c[i]
        a_new = b_prev + c_vals
        
        # Compute b[i] = a[i+1] * d[i] (using the old a[i+1])
        b_new = a_next * d_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s1213_kernel[grid](
        a, b, c, d, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )