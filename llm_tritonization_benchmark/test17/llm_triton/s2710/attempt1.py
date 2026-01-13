import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    x,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LEN_1D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional logic
    a_gt_b = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    new_a = tl.where(a_gt_b, a_vals + b_vals * d_vals, a_vals)
    
    # Inner condition for branch 1
    if LEN_1D > 10:
        c_branch1 = c_vals + d_vals * d_vals
    else:
        c_branch1 = d_vals * e_vals + 1.0
    
    # Branch 2: a[i] <= b[i]
    new_b = tl.where(~a_gt_b, a_vals + e_vals * e_vals, b_vals)
    
    # Inner condition for branch 2
    x_gt_zero = x > 0.0
    if x_gt_zero:
        c_branch2 = a_vals + d_vals * d_vals
    else:
        c_branch2 = c_vals + e_vals * e_vals
    
    # Select final c values based on main condition
    new_c = tl.where(a_gt_b, c_branch1, c_branch2)
    
    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)
    tl.store(c_ptr + offsets, new_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.numel()
    LEN_1D = n_elements
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        LEN_1D=LEN_1D,
    )