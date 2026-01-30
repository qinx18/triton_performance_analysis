import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + indices, mask=mask)
    b = tl.load(b_ptr + indices, mask=mask)
    c = tl.load(c_ptr + indices, mask=mask)
    d = tl.load(d_ptr + indices, mask=mask)
    e = tl.load(e_ptr + indices, mask=mask)
    
    # Conditional computation
    condition = a > 0.0
    
    # Compute b update for condition == False
    b_new = tl.where(condition, b, -b + d * e)
    
    # Compute c update for condition == True
    c_new = tl.where(condition, -c + d * e, c)
    
    # Final computation: a = b + c * d (using updated b and c)
    a_new = b_new + c_new * d
    
    # Store results
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, b_new, mask=mask)
    tl.store(c_ptr + indices, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c