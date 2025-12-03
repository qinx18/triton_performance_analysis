import torch
import triton
import triton.language as tl

@triton.jit
def s274_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load input values
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # First computation: a[i] = c[i] + e[i] * d[i]
    a_vals = c_vals + e_vals * d_vals
    
    # Conditional computation
    condition = a_vals > 0.0
    # If a[i] > 0: b[i] = a[i] + b[i], a[i] remains unchanged
    # If a[i] <= 0: a[i] = d[i] * e[i], b[i] remains unchanged
    b_new = tl.where(condition, a_vals + b_vals, b_vals)
    a_new = tl.where(condition, a_vals, d_vals * e_vals)
    
    # Store results
    tl.store(a_ptr + current_offsets, a_new, mask=mask)
    tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s274_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s274_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)