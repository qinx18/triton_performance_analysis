import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    b_val = tl.load(b_ptr + offsets, mask=mask)
    a_val = tl.load(a_ptr + offsets, mask=mask)
    c_val = tl.load(c_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    e_val = tl.load(e_ptr + offsets, mask=mask)
    
    # Load c[i+1] for the second branch (need to handle bounds)
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)  # Allow one more for c[i+1]
    c_plus_1_val = tl.load(c_ptr + offsets_plus_1, mask=mask_plus_1, other=0.0)
    
    # Condition: b[i] < 0
    condition = b_val < 0.0
    
    # Branch 1: a[i] = c[i] + d[i] * e[i] (when b[i] >= 0)
    new_a_val = c_val + d_val * e_val
    
    # Branch 2: c[i+1] = a[i] + d[i] * d[i] (when b[i] < 0)
    new_c_plus_1_val = a_val + d_val * d_val
    
    # Apply conditional updates
    # Update a[i] when condition is False (b[i] >= 0)
    final_a_val = tl.where(condition, a_val, new_a_val)
    tl.store(a_ptr + offsets, final_a_val, mask=mask)
    
    # Update c[i+1] when condition is True (b[i] < 0)
    final_c_plus_1_val = tl.where(condition, new_c_plus_1_val, c_plus_1_val)
    tl.store(c_ptr + offsets_plus_1, final_c_plus_1_val, mask=mask_plus_1)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, c