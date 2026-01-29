import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, x, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n
    
    a_val = tl.load(a_ptr + idx, mask=mask)
    b_val = tl.load(b_ptr + idx, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # if (a[i] > b[i])
    cond1 = a_val > b_val
    
    # True branch: a[i] > b[i]
    a_new_true = a_val + b_val * d_val
    c_new_true1 = c_val + d_val * d_val  # LEN_1D > 10 case (always true for typical size)
    c_new_true2 = d_val * e_val + 1.0    # LEN_1D <= 10 case
    # Since LEN_1D is typically 32000 > 10, we use c_new_true1
    c_new_true = tl.where(n > 10, c_new_true1, c_new_true2)
    
    # False branch: a[i] <= b[i]
    b_new_false = a_val + e_val * e_val
    c_new_false1 = a_val + d_val * d_val  # x > 0 case
    c_new_false2 = c_val + e_val * e_val  # x <= 0 case
    c_new_false = tl.where(x > 0.0, c_new_false1, c_new_false2)
    
    # Apply conditional updates
    a_new = tl.where(cond1, a_new_true, a_val)
    b_new = tl.where(cond1, b_val, b_new_false)
    c_new = tl.where(cond1, c_new_true, c_new_false)
    
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)
    tl.store(c_ptr + idx, c_new, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2710_kernel[grid](a, b, c, d, e, n, float(x), BLOCK_SIZE)