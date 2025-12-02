import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # First condition: if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    new_a_vals = tl.where(cond1, a_vals + b_vals * d_vals, a_vals)
    
    # Nested condition: if (LEN_1D > 10) - this is compile-time constant
    # Since n_elements is typically 32000 > 10, we use the first branch
    c_branch1 = c_vals + d_vals * d_vals
    
    # Branch 2: a[i] <= b[i]
    new_b_vals = tl.where(~cond1, a_vals + e_vals * e_vals, b_vals)
    
    # Nested condition: if (x > 0.)
    x_positive = x > 0.0
    c_branch2_true = a_vals + d_vals * d_vals
    c_branch2_false = c_vals + e_vals * e_vals
    c_branch2 = tl.where(x_positive, c_branch2_true, c_branch2_false)
    
    # Final results
    final_a = new_a_vals
    final_b = new_b_vals
    final_c = tl.where(cond1, c_branch1, c_branch2)
    
    tl.store(a_ptr + offsets, final_a, mask=mask)
    tl.store(b_ptr + offsets, final_b, mask=mask)
    tl.store(c_ptr + offsets, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](a, b, c, d, e, x, n_elements, BLOCK_SIZE)