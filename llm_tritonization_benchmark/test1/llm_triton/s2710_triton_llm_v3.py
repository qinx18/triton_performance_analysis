import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, len_1d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < len_1d
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional: if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    a_new_branch1 = a_vals + b_vals * d_vals
    
    # Nested conditional: if (LEN_1D > 10)
    len_1d_gt_10 = len_1d > 10
    if len_1d_gt_10:
        c_new_branch1 = c_vals + d_vals * d_vals
    else:
        c_new_branch1 = d_vals * e_vals + 1.0
    
    # Branch 2: a[i] <= b[i]
    b_new_branch2 = a_vals + e_vals * e_vals
    
    # Nested conditional: if (x > 0.)
    x_gt_zero = x > 0.0
    if x_gt_zero:
        c_new_branch2 = a_vals + d_vals * d_vals
    else:
        c_new_branch2 = c_vals + e_vals * e_vals
    
    # Select based on main condition
    a_final = tl.where(cond1, a_new_branch1, a_vals)
    b_final = tl.where(cond1, b_vals, b_new_branch2)
    c_final = tl.where(cond1, c_new_branch1, c_new_branch2)
    
    # Store results
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(b_ptr + offsets, b_final, mask=mask)
    tl.store(c_ptr + offsets, c_final, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    len_1d = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_1d, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e, 
        x, len_1d, 
        BLOCK_SIZE=BLOCK_SIZE
    )