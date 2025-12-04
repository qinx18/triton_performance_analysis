import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load c[i-1] values
        c_prev_offsets = current_offsets - 1
        c_prev_mask = mask & (c_prev_offsets >= 0)
        c_prev_vals = tl.load(c_ptr + c_prev_offsets, mask=c_prev_mask)
        
        # t = a[i] + b[i]
        t = a_vals + b_vals
        
        # a[i] = t + c[i-1]
        a_new = t + c_prev_vals
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # t = c[i] * d[i]
        t = c_vals * d_vals
        
        # c[i] = t
        tl.store(c_ptr + current_offsets, t, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s261_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )