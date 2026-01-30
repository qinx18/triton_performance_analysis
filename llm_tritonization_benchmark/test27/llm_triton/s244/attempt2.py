import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid indices (0 to n_elements-2)
    mask = idx < (n_elements - 1)
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_next_vals = tl.load(a_ptr + idx + 1, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + idx, b_new_vals, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (using updated b[i])
    a_next_new_vals = b_new_vals + a_next_vals * d_vals
    tl.store(a_ptr + idx + 1, a_next_new_vals, mask=mask)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    
    # Handle edge case
    if N <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )