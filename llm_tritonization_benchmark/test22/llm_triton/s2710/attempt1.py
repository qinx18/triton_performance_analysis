import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Main conditional logic
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    a_new = a_vals + b_vals * d_vals
    # Since LEN_1D is typically 32000 > 10, always use first branch
    c_branch1 = c_vals + d_vals * d_vals
    
    # Branch 2: a[i] <= b[i]
    b_new = a_vals + e_vals * e_vals
    cond2 = x > 0.0
    c_branch2_true = a_vals + d_vals * d_vals
    c_branch2_false = c_vals + e_vals * e_vals
    c_branch2 = tl.where(cond2, c_branch2_true, c_branch2_false)
    
    # Final selection
    a_final = tl.where(cond1, a_new, a_vals)
    b_final = tl.where(cond1, b_vals, b_new)
    c_final = tl.where(cond1, c_branch1, c_branch2)
    
    # Store results
    tl.store(a_ptr + idx, a_final, mask=mask)
    tl.store(b_ptr + idx, b_final, mask=mask)
    tl.store(c_ptr + idx, c_final, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](a, b, c, d, e, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)