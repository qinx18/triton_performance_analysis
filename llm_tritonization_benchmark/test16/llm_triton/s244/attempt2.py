import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i] (using original b[i])
        new_b_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i] (using updated b[i])
        offset_plus_one = current_offsets + 1
        mask_plus_one = offset_plus_one <= n_elements
        
        a_copy_vals = tl.load(a_copy_ptr + offset_plus_one, mask=mask_plus_one)
        result_vals = new_b_vals + a_copy_vals * d_vals
        tl.store(a_ptr + offset_plus_one, result_vals, mask=mask_plus_one)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s244_kernel[grid](
        a, b, c, d, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )