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
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: b[i] < 0
    condition = b_vals < 0.0
    
    # Branch 1: a[i] = c[i] + d[i] * e[i] (when b[i] >= 0)
    branch1_result = c_vals + d_vals * e_vals
    
    # Branch 2: c[i+1] = a[i] + d[i] * d[i] (when b[i] < 0)
    branch2_result = a_vals + d_vals * d_vals
    
    # Store a[i] for branch 1
    branch1_mask = mask & (~condition)
    tl.store(a_ptr + idx, branch1_result, mask=branch1_mask)
    
    # Store c[i+1] for branch 2 - need to check bounds for i+1
    c_next_idx = idx + 1
    c_next_mask = c_next_idx < (a_ptr.shape[0])  # Check against original array size
    branch2_mask = mask & condition & c_next_mask
    tl.store(c_ptr + c_next_idx, branch2_result, mask=branch2_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )