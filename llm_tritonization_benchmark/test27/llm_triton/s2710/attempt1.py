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
    
    # Main conditional: a[i] > b[i]
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i] is true
    a_new1 = a_vals + b_vals * d_vals
    # Since LEN_1D is typically 32000 > 10, always use first branch
    c_new1 = c_vals + d_vals * d_vals
    
    # Branch 2: a[i] > b[i] is false
    b_new2 = a_vals + e_vals * e_vals
    # Check if x > 0
    x_positive = x > 0.0
    c_new2_pos = a_vals + d_vals * d_vals
    c_new2_neg = c_vals + e_vals * e_vals
    c_new2 = tl.where(x_positive, c_new2_pos, c_new2_neg)
    
    # Apply main conditional
    a_final = tl.where(cond1, a_new1, a_vals)
    b_final = tl.where(cond1, b_vals, b_new2)
    c_final = tl.where(cond1, c_new1, c_new2)
    
    # Store results
    tl.store(a_ptr + idx, a_final, mask=mask)
    tl.store(b_ptr + idx, b_final, mask=mask)
    tl.store(c_ptr + idx, c_final, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](a, b, c, d, e, x, n_elements, BLOCK_SIZE)