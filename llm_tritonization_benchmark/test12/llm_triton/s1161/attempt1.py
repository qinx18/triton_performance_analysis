import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Conditional computation
    condition = c_vals < 0.0
    
    # Path 1: c[i] >= 0, compute a[i] = c[i] + d[i] * e[i]
    a_result = c_vals + d_vals * e_vals
    
    # Path 2: c[i] < 0, compute b[i] = a[i] + d[i] * d[i]
    b_result = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # When condition is False (c[i] >= 0), update a
    tl.store(a_ptr + indices, a_result, mask=mask & ~condition)
    
    # When condition is True (c[i] < 0), update b
    tl.store(b_ptr + indices, b_result, mask=mask & condition)

def s1161_triton(a, b, c, d, e):
    n_elements = c.numel() - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )