import triton
import triton.language as tl
import torch

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Step 1: Process else branch (b[i] < 0) - writes c[i+1]
    else_mask = (b_vals < 0.0) & mask
    c_next_offsets = offsets + 1
    c_next_mask = c_next_offsets < (n_elements + 1)
    c_next_result = a_vals + d_vals * d_vals
    tl.store(c_ptr + c_next_offsets, c_next_result, mask=else_mask & c_next_mask)
    
    # Step 2: Process if branch (b[i] >= 0) - writes a[i]
    if_mask = (b_vals >= 0.0) & mask
    # Need to reload c_vals in case they were modified by else branch
    c_vals_updated = tl.load(c_ptr + offsets, mask=mask)
    a_result = c_vals_updated + d_vals * e_vals
    tl.store(a_ptr + offsets, a_result, mask=if_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )