import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Condition: c[i] < 0
        condition = c_vals < 0.0
        
        # Branch 1: a[i] = c[i] + d[i] * e[i] (when c[i] >= 0)
        new_a_vals = c_vals + d_vals * e_vals
        
        # Branch 2: b[i] = a[i] + d[i] * d[i] (when c[i] < 0)
        new_b_vals = a_vals + d_vals * d_vals
        
        # Store results based on condition
        # When c[i] >= 0: update a[i]
        a_update_mask = mask & (~condition)
        tl.store(a_ptr + current_offsets, new_a_vals, mask=a_update_mask)
        
        # When c[i] < 0: update b[i]
        b_update_mask = mask & condition
        tl.store(b_ptr + current_offsets, new_b_vals, mask=b_update_mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    s1161_kernel[(1,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE
    )