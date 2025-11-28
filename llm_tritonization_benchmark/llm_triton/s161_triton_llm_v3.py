import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Load c[i+1] values for the else branch
    c_next_offsets = offsets + 1
    c_next_mask = c_next_offsets < (n_elements + 1)
    c_next_vals = tl.load(c_ptr + c_next_offsets, mask=c_next_mask)
    
    # Conditional computation
    condition = b_vals < 0.0
    
    # If branch: a[i] = c[i] + d[i] * e[i]
    a_if = c_vals + d_vals * e_vals
    
    # Else branch: c[i+1] = a[i] + d[i] * d[i]
    c_else = a_vals + d_vals * d_vals
    
    # Update a[i] for if branch
    new_a = tl.where(condition, a_vals, a_if)
    tl.store(a_ptr + offsets, new_a, mask=mask)
    
    # Update c[i+1] for else branch
    new_c_next = tl.where(condition, c_else, c_next_vals)
    tl.store(c_ptr + c_next_offsets, new_c_next, mask=c_next_mask)

def s161_triton(a, b, c, d, e):
    n_elements = len(a) - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)