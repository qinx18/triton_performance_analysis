import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block range
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (i < N-1)
    mask = indices < (N - 1)
    
    # Load arrays
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    a_next_vals = tl.load(a_ptr + indices + 1, mask=mask)
    
    # Execute statements S0 and S1 for all iterations
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new = c_vals + b_vals
    tl.store(b_ptr + indices, b_new, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] - only for last iteration (i == N-2)
    last_iter_mask = mask & (indices == (N - 2))
    if tl.sum(last_iter_mask.to(tl.int32)) > 0:
        a_next_new = b_new + a_next_vals * d_vals
        tl.store(a_ptr + indices + 1, a_next_new, mask=last_iter_mask)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    s244_kernel[grid](a, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE)