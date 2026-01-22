import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load input data
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute j values - each i maps to j starting from 0
    j_vals = i_offsets
    
    # First store: a[j] = b[i] + d[i] * e[i]
    result1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_vals, result1, mask=mask)
    
    # For conditional store: a[j+1] = c[i] + d[i] * e[i] when c[i] > 0
    cond_mask = mask & (c_vals > 0.0)
    j_vals_cond = j_vals + 1
    result2 = c_vals + d_vals * e_vals
    tl.store(a_ptr + j_vals_cond, result2, mask=cond_mask)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )