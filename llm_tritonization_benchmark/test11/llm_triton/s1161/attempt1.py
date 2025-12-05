import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load input arrays
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Conditional computation: if c[i] < 0, compute b[i], else compute a[i]
    condition = c_vals < 0.0
    
    # a[i] = c[i] + d[i] * e[i] (when c[i] >= 0)
    a_result = c_vals + d_vals * e_vals
    
    # b[i] = a[i] + d[i] * d[i] (when c[i] < 0)
    b_result = a_vals + d_vals * d_vals
    
    # Store results conditionally
    tl.store(a_ptr + indices, tl.where(condition, a_vals, a_result), mask=mask)
    tl.store(b_ptr + indices, tl.where(condition, b_result, tl.load(b_ptr + indices, mask=mask)), mask=mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b