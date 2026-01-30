import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Compute j indices (j starts at -1, then increments)
    j_even = 2 * i_offsets      # j values for first assignment
    j_odd = 2 * i_offsets + 1   # j values for second assignment
    
    # Compute values
    a_even = b_vals + c_vals * d_vals
    a_odd = b_vals + d_vals * e_vals
    
    # Store results
    tl.store(a_ptr + j_even, a_even, mask=mask)
    tl.store(a_ptr + j_odd, a_odd, mask=mask)

def s127_triton(a, b, c, d, e):
    n = a.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a