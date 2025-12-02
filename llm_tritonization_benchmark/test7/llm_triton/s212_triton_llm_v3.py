import torch
import triton
import triton.language as tl

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load current elements
    a_vals = tl.load(a_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # First statement: a[i] *= c[i]
    a_vals = a_vals * c_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Load a[i+1] for second statement
    indices_plus_1 = indices + 1
    mask_plus_1 = indices_plus_1 < (n_elements + 1)
    a_plus_1_vals = tl.load(a_ptr + indices_plus_1, mask=mask_plus_1)
    
    # Second statement: b[i] += a[i + 1] * d[i]
    # Only compute where both current index is valid AND i+1 is valid
    valid_mask = mask & mask_plus_1
    b_update = a_plus_1_vals * d_vals
    b_vals = tl.where(valid_mask, b_vals + b_update, b_vals)
    tl.store(b_ptr + indices, b_vals, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )