import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load input values
    a_val = tl.load(a_ptr + idx, mask=mask)
    b_val = tl.load(b_ptr + idx, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: a[i] > 0
    condition = a_val > 0.0
    
    # Compute both branches
    # Branch 1 (else): b[i] = -b[i] + d[i] * e[i]
    b_new_else = -b_val + d_val * e_val
    c_new_else = c_val  # c unchanged in else branch
    
    # Branch 2 (if): c[i] = -c[i] + d[i] * e[i]
    b_new_if = b_val  # b unchanged in if branch
    c_new_if = -c_val + d_val * e_val
    
    # Select based on condition
    b_new = tl.where(condition, b_new_if, b_new_else)
    c_new = tl.where(condition, c_new_if, c_new_else)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d_val
    
    # Store results
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)
    tl.store(c_ptr + idx, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )