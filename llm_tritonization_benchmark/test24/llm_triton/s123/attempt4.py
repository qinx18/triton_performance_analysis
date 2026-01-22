import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < (n // 2)
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Compute j for each i: j starts at i (since j starts at -1, then ++j before first use)
    j_offsets = i_offsets
    
    # First write: a[j] = b[i] + d[i] * e[i]
    first_vals = b_vals + d_vals * e_vals
    first_mask = mask & (j_offsets < n)
    tl.store(a_ptr + j_offsets, first_vals, mask=first_mask)
    
    # Second conditional write: a[j+1] = c[i] + d[i] * e[i] if c[i] > 0
    cond = c_vals > 0.0
    second_vals = c_vals + d_vals * e_vals
    j_next = j_offsets + 1
    second_mask = mask & cond & (j_next < n)
    tl.store(a_ptr + j_next, second_vals, mask=second_mask)

def s123_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n // 2, BLOCK_SIZE),)
    
    s123_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a