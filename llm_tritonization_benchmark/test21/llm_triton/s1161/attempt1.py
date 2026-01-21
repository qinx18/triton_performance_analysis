import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Create condition mask for c[i] < 0
    neg_mask = c_vals < 0.0
    pos_mask = ~neg_mask
    
    # For positive c[i]: a[i] = c[i] + d[i] * e[i]
    a_new = tl.where(pos_mask, c_vals + d_vals * e_vals, a_vals)
    
    # For negative c[i]: b[i] = a[i] + d[i] * d[i]
    b_new = a_vals + d_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=(mask & neg_mask))

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1161_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)