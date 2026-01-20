import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, x, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load all arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional: if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # First branch: a[i] > b[i]
    a_new = tl.where(cond1, a_vals + b_vals * d_vals, a_vals)
    
    # Nested conditional for first branch: if (LEN_1D > 10)
    # Since n is the actual array length, use n > 10
    len_cond = n > 10
    if len_cond:
        c_branch1 = c_vals + d_vals * d_vals
    else:
        c_branch1 = d_vals * e_vals + 1.0
    
    # Second branch: a[i] <= b[i]
    b_new = tl.where(cond1, b_vals, a_vals + e_vals * e_vals)
    
    # Nested conditional for second branch: if (x > 0.)
    x_cond = x > 0.0
    if x_cond:
        c_branch2 = a_vals + d_vals * d_vals
    else:
        c_branch2 = c_vals + e_vals * e_vals
    
    # Select final c values based on main condition
    c_new = tl.where(cond1, c_branch1, c_branch2)
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_new, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e, n, x, BLOCK_SIZE=BLOCK_SIZE
    )