import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Main loop mask - process i = 0 to n-2
    mask = idx < (n - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # S1: a[i] = b[i] + c[i] for all iterations
    s1_result = b_vals + c_vals
    tl.store(a_ptr + idx, s1_result, mask=mask)
    
    # S0: a[i+1] = b[i] + e[i] only for last iteration (i = n-2)
    last_iter_mask = mask & (idx == (n - 2))
    s0_result = b_vals + e_vals
    tl.store(a_ptr + idx + 1, s0_result, mask=last_iter_mask)

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )