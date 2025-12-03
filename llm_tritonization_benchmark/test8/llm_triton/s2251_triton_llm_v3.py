import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        a_vals = s * e_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        s_new = b_vals + c_vals
        
        a_vals_reloaded = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_new = a_vals_reloaded + d_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
        
        if block_start + BLOCK_SIZE - 1 < n_elements:
            s = tl.load(b_ptr + c_ptr + (block_start + BLOCK_SIZE - 1))
        else:
            last_valid_idx = n_elements - 1
            if last_valid_idx >= 0:
                s = tl.load(b_ptr + last_valid_idx) + tl.load(c_ptr + last_valid_idx)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )