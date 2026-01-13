import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Get block offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (0 to N-2, since we process i < N-1)
    mask = indices < (N - 1)
    
    # Load arrays for current indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Load a[i+1] for third statement
    a_next_vals = tl.load(a_ptr + indices + 1, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + indices, b_new_vals, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (using updated b[i])
    a_next_new_vals = b_new_vals + a_next_vals * d_vals
    tl.store(a_ptr + indices + 1, a_next_new_vals, mask=mask)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    
    # We need to process indices 0 to N-2 (since loop condition is i < N-1)
    n_elements = N - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )