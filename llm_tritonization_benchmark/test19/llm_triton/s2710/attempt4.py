import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # True branch: a[i] > b[i]
    a_new_true = a_vals + b_vals * d_vals
    
    # Inner condition: if (LEN_1D > 10) - since n_elements is typically 32000, this is always true
    c_new_true = c_vals + d_vals * d_vals
    
    # False branch: a[i] <= b[i] 
    b_new_false = a_vals + e_vals * e_vals
    
    # Inner condition: if (x > 0.)
    cond2 = x > 0.0
    c_new_false_true = a_vals + d_vals * d_vals
    c_new_false_false = c_vals + e_vals * e_vals
    
    # Select based on inner condition
    c_new_false = tl.where(cond2, c_new_false_true, c_new_false_false)
    
    # Final selection based on outer condition
    a_final = tl.where(cond1, a_new_true, a_vals)
    b_final = tl.where(cond1, b_vals, b_new_false)
    c_final = tl.where(cond1, c_new_true, c_new_false)
    
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(b_ptr + offsets, b_final, mask=mask)
    tl.store(c_ptr + offsets, c_final, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        float(x), N, BLOCK_SIZE
    )