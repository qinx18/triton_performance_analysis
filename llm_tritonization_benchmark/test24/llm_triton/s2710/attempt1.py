import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Main conditional: if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    a_new_1 = a_vals + b_vals * d_vals
    # Since LEN_1D > 10 (typically 32000), always take first branch
    c_new_1 = c_vals + d_vals * d_vals
    
    # Branch 2: a[i] <= b[i]
    b_new_2 = a_vals + e_vals * e_vals
    # Nested conditional: if (x > 0.)
    cond2 = x > 0.0
    c_new_2a = a_vals + d_vals * d_vals  # x > 0
    c_new_2b = c_vals + e_vals * e_vals  # x <= 0
    c_new_2 = tl.where(cond2, c_new_2a, c_new_2b)
    
    # Combine results based on main condition
    a_final = tl.where(cond1, a_new_1, a_vals)
    b_final = tl.where(cond1, b_vals, b_new_2)
    c_final = tl.where(cond1, c_new_1, c_new_2)
    
    # Store results
    tl.store(a_ptr + indices, a_final, mask=mask)
    tl.store(b_ptr + indices, b_final, mask=mask)
    tl.store(c_ptr + indices, c_final, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e, 
        x, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )