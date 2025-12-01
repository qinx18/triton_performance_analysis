import triton
import triton.language as tl
import torch

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
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Load c[i+1] for the case when b[i] < 0
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)  # Allow reading c[i+1]
    c_vals_plus_1 = tl.load(c_ptr + offsets_plus_1, mask=mask_plus_1)
    
    # Condition: b[i] < 0
    condition = b_vals < 0.0
    
    # Branch 1: b[i] >= 0 => a[i] = c[i] + d[i] * e[i]
    new_a_vals = c_vals + d_vals * e_vals
    
    # Branch 2: b[i] < 0 => c[i+1] = a[i] + d[i] * d[i]
    new_c_vals_plus_1 = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # Update a[i] when b[i] >= 0
    tl.store(a_ptr + offsets, new_a_vals, mask=mask & ~condition)
    
    # Update c[i+1] when b[i] < 0
    tl.store(c_ptr + offsets_plus_1, new_c_vals_plus_1, mask=mask_plus_1 & condition)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )