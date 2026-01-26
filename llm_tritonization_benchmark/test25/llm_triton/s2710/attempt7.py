import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Main condition: if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # True branch: a[i] > b[i]
    new_a_true = a_vals + b_vals * d_vals
    
    # Inner condition for true branch: if (LEN_1D > 10)
    new_c_true = tl.where(n > 10, 
                         c_vals + d_vals * d_vals,
                         d_vals * e_vals + 1.0)
    
    # False branch: a[i] <= b[i]
    new_b_false = a_vals + e_vals * e_vals
    
    # Inner condition for false branch: if (x > 0)
    new_c_false = tl.where(x > 0.0,
                          a_vals + d_vals * d_vals,
                          c_vals + e_vals * e_vals)
    
    # Select results based on main condition
    final_a = tl.where(cond1, new_a_true, a_vals)
    final_b = tl.where(cond1, b_vals, new_b_false)
    final_c = tl.where(cond1, new_c_true, new_c_false)
    
    # Store results
    tl.store(a_ptr + idx, final_a, mask=mask)
    tl.store(b_ptr + idx, final_b, mask=mask)
    tl.store(c_ptr + idx, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2710_kernel[grid](a, b, c, d, e, x, n, BLOCK_SIZE)