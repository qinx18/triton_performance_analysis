import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Condition: c[i] < 0
        condition = c_vals < 0.0
        
        # Path 1: c[i] >= 0 -> a[i] = c[i] + d[i] * e[i]
        path1_result = c_vals + d_vals * e_vals
        
        # Path 2: c[i] < 0 -> b[i] = a[i] + d[i] * d[i]
        path2_result = a_vals + d_vals * d_vals
        
        # Update a for path 1 (when c[i] >= 0)
        a_update_mask = mask & (~condition)
        tl.store(a_ptr + current_offsets, 
                tl.where(~condition, path1_result, a_vals), 
                mask=mask)
        
        # Update b for path 2 (when c[i] < 0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        tl.store(b_ptr + current_offsets,
                tl.where(condition, path2_result, b_vals),
                mask=mask)

def s1161_triton(a, b, c, d, e):
    n_elements = c.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )