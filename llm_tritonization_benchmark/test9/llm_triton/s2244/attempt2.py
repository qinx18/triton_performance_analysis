import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i], c[i], and e[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute a[i+1] = b[i] + e[i]
        a_i_plus_1_vals = b_vals + e_vals
        next_offsets = current_offsets + 1
        next_mask = mask
        tl.store(a_ptr + next_offsets, a_i_plus_1_vals, mask=next_mask)
        
        # Compute a[i] = b[i] + c[i] 
        a_i_vals = b_vals + c_vals
        tl.store(a_ptr + current_offsets, a_i_vals, mask=mask)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )