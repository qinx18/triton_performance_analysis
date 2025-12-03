import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Check condition: c[i] < 0
    condition = c_vals < 0.0
    
    # Compute both branches
    # Branch 1: a[i] = c[i] + d[i] * e[i]
    branch1_result = c_vals + d_vals * e_vals
    
    # Branch 2: b[i] = a[i] + d[i] * d[i]
    branch2_result = a_vals + d_vals * d_vals
    
    # Apply conditional execution
    # If c[i] < 0, execute branch 2 (update b), otherwise execute branch 1 (update a)
    new_a_vals = tl.where(condition, a_vals, branch1_result)
    new_b_vals = tl.where(condition, branch2_result, tl.load(b_ptr + indices, mask=mask))
    
    # Store results
    tl.store(a_ptr + indices, new_a_vals, mask=mask)
    tl.store(b_ptr + indices, new_b_vals, mask=mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )