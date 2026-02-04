import triton
import triton.language as tl
import torch

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
    
    # Load all arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute condition: a[i] > 0
    condition = a_vals > 0.0
    
    # Compute both branches
    # Branch 1 (a[i] <= 0): b[i] = -b[i] + d[i] * e[i]
    b_branch1 = -b_vals + d_vals * e_vals
    c_branch1 = c_vals  # c unchanged in this branch
    
    # Branch 2 (a[i] > 0): c[i] = -c[i] + d[i] * e[i]
    b_branch2 = b_vals  # b unchanged in this branch
    c_branch2 = -c_vals + d_vals * e_vals
    
    # Select based on condition
    b_new = tl.where(condition, b_branch2, b_branch1)
    c_new = tl.where(condition, c_branch2, c_branch1)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d_vals
    
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