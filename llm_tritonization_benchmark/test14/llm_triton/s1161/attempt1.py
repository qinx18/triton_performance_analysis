import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    a_val = tl.load(a_ptr + idx, mask=mask)
    
    # Check condition: if c[i] < 0
    cond = c_val < 0.0
    
    # If c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    # If c[i] < 0: b[i] = a[i] + d[i] * d[i]
    
    # Compute both branches
    branch_if = c_val + d_val * e_val  # a[i] = c[i] + d[i] * e[i]
    branch_else = a_val + d_val * d_val  # b[i] = a[i] + d[i] * d[i]
    
    # Apply conditional updates
    new_a = tl.where(cond, a_val, branch_if)  # Update a only when c[i] >= 0
    new_b = tl.where(cond, branch_else, tl.load(b_ptr + idx, mask=mask))  # Update b only when c[i] < 0
    
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )