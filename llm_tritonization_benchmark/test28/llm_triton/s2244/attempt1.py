import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Main computation for i = 0 to N-2
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process main iterations
    idx = block_start + offsets
    mask = idx < (N - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # S1: a[i] = b[i] + c[i] (executes for all valid indices)
    result1 = b_vals + c_vals
    tl.store(a_ptr + idx, result1, mask=mask)
    
    # S0: a[i+1] = b[i] + e[i] (only for last iteration i = N-2)
    last_iter_mask = mask & (idx == (N - 2))
    if tl.any(last_iter_mask):
        result0 = b_vals + e_vals
        tl.store(a_ptr + idx + 1, result0, mask=last_iter_mask)

def s2244_triton(a, b, c, e):
    N = a.shape[0]
    
    if N <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](a, b, c, e, N, BLOCK_SIZE)