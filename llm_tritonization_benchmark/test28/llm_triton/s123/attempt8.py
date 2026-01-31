import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load data for current block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate common expression
    de_vals = d_vals * e_vals
    
    # Store a[2*i] = b[i] + d[i] * e[i]
    j_offsets = 2 * i_offsets
    val1 = b_vals + de_vals
    tl.store(a_ptr + j_offsets, val1, mask=mask)
    
    # Store a[2*i+1] = c[i] + d[i] * e[i] only if c[i] > 0
    condition = c_vals > 0.0
    j_offsets_plus1 = j_offsets + 1
    val2 = c_vals + de_vals
    combined_mask = mask & condition
    tl.store(a_ptr + j_offsets_plus1, val2, mask=combined_mask)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )