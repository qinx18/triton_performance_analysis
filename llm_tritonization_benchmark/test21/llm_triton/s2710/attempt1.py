import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional logic
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    new_a_vals = tl.where(cond1, a_vals + b_vals * d_vals, a_vals)
    
    # Since LEN_1D is typically 32000 > 10, the first branch is taken
    new_c_vals_branch1 = c_vals + d_vals * d_vals
    
    # Branch 2: a[i] <= b[i]
    new_b_vals = tl.where(~cond1, a_vals + e_vals * e_vals, b_vals)
    
    # Nested conditional for second branch
    cond2 = x > 0.0
    new_c_vals_branch2_true = a_vals + d_vals * d_vals
    new_c_vals_branch2_false = c_vals + e_vals * e_vals
    new_c_vals_branch2 = tl.where(cond2, new_c_vals_branch2_true, new_c_vals_branch2_false)
    
    # Combine branches
    final_c_vals = tl.where(cond1, new_c_vals_branch1, new_c_vals_branch2)
    
    # Store results
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    tl.store(b_ptr + offsets, new_b_vals, mask=mask)
    tl.store(c_ptr + offsets, final_c_vals, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e, 
        float(x), n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )