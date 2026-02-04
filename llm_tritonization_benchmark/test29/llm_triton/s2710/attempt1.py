import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Main condition: a[i] > b[i]
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    a_new_1 = a_vals + b_vals * d_vals
    c_new_1 = tl.where(n_elements > 10, c_vals + d_vals * d_vals, d_vals * e_vals + 1.0)
    
    # Branch 2: a[i] <= b[i]
    b_new_2 = a_vals + e_vals * e_vals
    c_new_2 = tl.where(x > 0.0, a_vals + d_vals * d_vals, c_vals + e_vals * e_vals)
    
    # Select results based on main condition
    a_result = tl.where(cond1, a_new_1, a_vals)
    b_result = tl.where(cond1, b_vals, b_new_2)
    c_result = tl.where(cond1, c_new_1, c_new_2)
    
    # Store results
    tl.store(a_ptr + idx, a_result, mask=mask)
    tl.store(b_ptr + idx, b_result, mask=mask)
    tl.store(c_ptr + idx, c_result, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        x, n_elements, BLOCK_SIZE
    )