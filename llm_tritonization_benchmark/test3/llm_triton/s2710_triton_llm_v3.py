import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    x,
    len_1d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < len_1d
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional logic
    condition1 = a_vals > b_vals
    
    # Branch 1: if (a[i] > b[i])
    new_a = a_vals + b_vals * d_vals
    
    # Nested condition for branch 1
    if len_1d > 10:
        new_c_branch1 = c_vals + d_vals * d_vals
    else:
        new_c_branch1 = d_vals * e_vals + 1.0
    
    # Branch 2: else
    new_b = a_vals + e_vals * e_vals
    
    # Nested condition for branch 2
    if x > 0.0:
        new_c_branch2 = a_vals + d_vals * d_vals
    else:
        new_c_branch2 = c_vals + e_vals * e_vals
    
    # Select results based on condition
    final_a = tl.where(condition1, new_a, a_vals)
    final_b = tl.where(condition1, b_vals, new_b)
    final_c = tl.where(condition1, new_c_branch1, new_c_branch2)
    
    # Store results
    tl.store(a_ptr + offsets, final_a, mask=mask)
    tl.store(b_ptr + offsets, final_b, mask=mask)
    tl.store(c_ptr + offsets, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    len_1d = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_1d, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        float(x),
        len_1d,
        BLOCK_SIZE,
    )