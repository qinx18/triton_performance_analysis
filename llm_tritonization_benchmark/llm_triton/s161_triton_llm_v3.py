import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    b = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Load c[i+1] and d[i+1] for the else branch (need to handle boundary)
    offsets_plus_one = offsets + 1
    mask_plus_one = offsets_plus_one < (n_elements + 1)  # Allow one extra element
    d_vals_plus_one = tl.load(d_ptr + offsets, mask=mask)  # d[i] for d[i]*d[i]
    
    # Condition: b[i] < 0
    condition = b < 0.0
    
    # Branch 1: b[i] >= 0 -> a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    
    # Branch 2: b[i] < 0 -> c[i+1] = a[i] + d[i] * d[i]
    c_new_val = a_vals + d_vals * d_vals
    
    # Update a[i] when b[i] >= 0
    a_result = tl.where(condition, a_vals, a_new)
    tl.store(a_ptr + offsets, a_result, mask=mask)
    
    # Update c[i+1] when b[i] < 0 (need to handle this carefully)
    # We need to write to c[i+1] position
    write_mask = mask & condition & (offsets + 1 < n_elements + 1)
    tl.store(c_ptr + offsets + 1, c_new_val, mask=write_mask)

def s161_triton(a, b, c, d, e):
    n_elements = len(a) - 1  # Loop goes to LEN_1D-1
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c