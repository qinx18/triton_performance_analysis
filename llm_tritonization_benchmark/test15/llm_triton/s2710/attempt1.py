import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr, LEN_1D: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Main conditional logic
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    a_new_branch1 = a_vals + b_vals * d_vals
    if LEN_1D > 10:
        c_new_branch1 = c_vals + d_vals * d_vals
    else:
        c_new_branch1 = d_vals * e_vals + 1.0
    
    # Branch 2: a[i] <= b[i]
    b_new_branch2 = a_vals + e_vals * e_vals
    cond2 = x > 0.0
    c_new_branch2_true = a_vals + d_vals * d_vals
    c_new_branch2_false = c_vals + e_vals * e_vals
    c_new_branch2 = tl.where(cond2, c_new_branch2_true, c_new_branch2_false)
    
    # Select final values based on main condition
    a_final = tl.where(cond1, a_new_branch1, a_vals)
    b_final = tl.where(cond1, b_vals, b_new_branch2)
    c_final = tl.where(cond1, c_new_branch1, c_new_branch2)
    
    # Store results
    tl.store(a_ptr + indices, a_final, mask=mask)
    tl.store(b_ptr + indices, b_final, mask=mask)
    tl.store(c_ptr + indices, c_final, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    LEN_1D = n_elements
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e, x, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE, LEN_1D=LEN_1D
    )