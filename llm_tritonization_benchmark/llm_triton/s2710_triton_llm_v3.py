import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    len_1d: tl.constexpr,
    x,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < len_1d
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional logic
    a_gt_b = a_vals > b_vals
    
    # If a[i] > b[i] branch
    new_a_vals = tl.where(a_gt_b, a_vals + b_vals * d_vals, a_vals)
    
    # Nested if for when a[i] > b[i]
    if len_1d > 10:
        c_vals_true_branch = c_vals + d_vals * d_vals
    else:
        c_vals_true_branch = d_vals * e_vals + 1.0
    
    # Else branch: b[i] = a[i] + e[i] * e[i]
    new_b_vals = tl.where(a_gt_b, b_vals, a_vals + e_vals * e_vals)
    
    # Nested if for else branch
    x_gt_zero = x > 0.0
    c_vals_false_branch = tl.where(x_gt_zero, a_vals + d_vals * d_vals, c_vals + e_vals * e_vals)
    
    # Final c values based on main condition
    new_c_vals = tl.where(a_gt_b, c_vals_true_branch, c_vals_false_branch)
    
    # Store results
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    tl.store(b_ptr + offsets, new_b_vals, mask=mask)
    tl.store(c_ptr + offsets, new_c_vals, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    len_1d = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_1d, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        len_1d,
        x,
        BLOCK_SIZE=BLOCK_SIZE,
    )