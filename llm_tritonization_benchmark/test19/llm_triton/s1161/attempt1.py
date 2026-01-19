import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute conditions
    c_negative = c_vals < 0.0
    c_non_negative = ~c_negative
    
    # Compute results for both branches
    # If c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    a_result = c_vals + d_vals * e_vals
    
    # If c[i] < 0: b[i] = a[i] + d[i] * d[i]
    b_result = a_vals + d_vals * d_vals
    
    # Store results based on conditions
    # Update a when c[i] >= 0
    tl.store(a_ptr + offsets, a_result, mask=mask & c_non_negative)
    
    # Update b when c[i] < 0
    tl.store(b_ptr + offsets, b_result, mask=mask & c_negative)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )