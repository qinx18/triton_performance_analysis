import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute condition: c[i] < 0.0
        condition = c_vals < 0.0
        
        # Path 1: c[i] >= 0.0 -> a[i] = c[i] + d[i] * e[i]
        a_result = c_vals + d_vals * e_vals
        
        # Path 2: c[i] < 0.0 -> b[i] = a[i] + d[i] * d[i]
        b_result = a_vals + d_vals * d_vals
        
        # Store results based on condition
        # For a: only store when c[i] >= 0.0
        a_mask = mask & (~condition)
        tl.store(a_ptr + current_offsets, a_result, mask=a_mask)
        
        # For b: only store when c[i] < 0.0
        b_mask = mask & condition
        tl.store(b_ptr + current_offsets, b_result, mask=b_mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D-1 as per original loop
    BLOCK_SIZE = 256
    
    s1161_kernel[(1,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE
    )