import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Conditional computation: if c[i] < 0, execute branch L20, else main branch
    condition = c_vals < 0.0
    
    # Main branch: a[i] = c[i] + d[i] * e[i]
    a_result = c_vals + d_vals * e_vals
    
    # Branch L20: b[i] = a[i] + d[i] * d[i]
    b_result = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # For main branch (c[i] >= 0): update a[i]
    tl.store(a_ptr + offsets, a_result, mask=mask & ~condition)
    
    # For L20 branch (c[i] < 0): update b[i]
    tl.store(b_ptr + offsets, b_result, mask=mask & condition)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )