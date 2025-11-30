import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    c_val = tl.load(c_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    e_val = tl.load(e_ptr + offsets, mask=mask)
    a_val = tl.load(a_ptr + offsets, mask=mask)
    
    # Condition check: c[i] < 0
    condition = c_val < 0.0
    
    # Branch 1: c[i] >= 0, compute a[i] = c[i] + d[i] * e[i]
    a_result = c_val + d_val * e_val
    
    # Branch 2: c[i] < 0, compute b[i] = a[i] + d[i] * d[i]
    b_result = a_val + d_val * d_val
    
    # Store results based on condition
    # For positive c values, update a
    tl.store(a_ptr + offsets, a_result, mask=mask & ~condition)
    
    # For negative c values, update b
    tl.store(b_ptr + offsets, b_result, mask=mask & condition)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )