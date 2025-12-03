import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    
    # Load c[i+1] for the second branch
    c_next_idx = idx + 1
    c_next_mask = c_next_idx < (n_elements + 1)
    
    # Condition: b[i] < 0
    condition = b_vals < 0.0
    
    # Branch 1: b[i] >= 0 -> a[i] = c[i] + d[i] * e[i]
    a_branch1 = c_vals + d_vals * e_vals
    
    # Branch 2: b[i] < 0 -> c[i+1] = a[i] + d[i] * d[i]
    c_branch2 = a_vals + d_vals * d_vals
    
    # Update a[i] when b[i] >= 0
    a_update_mask = mask & (~condition)
    a_result = tl.where(condition, a_vals, a_branch1)
    tl.store(a_ptr + idx, a_result, mask=mask)
    
    # Update c[i+1] when b[i] < 0
    c_update_mask = c_next_mask & tl.where(idx < n_elements, condition, False)
    tl.store(c_ptr + c_next_idx, c_branch2, mask=c_update_mask)

def s161_triton(a, b, c, d, e):
    n_elements = len(a) - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )