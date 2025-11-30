import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current elements
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Load a[i+1] elements for the second statement
    offsets_plus_one = offsets + 1
    mask_plus_one = offsets_plus_one < (n_elements + 1)  # Allow reading a[i+1]
    a_plus_one_vals = tl.load(a_ptr + offsets_plus_one, mask=mask_plus_one, other=0.0)
    
    # First statement: a[i] *= c[i]
    a_updated = a_vals * c_vals
    
    # Store updated a values
    tl.store(a_ptr + offsets, a_updated, mask=mask)
    
    # Second statement: b[i] += a[i + 1] * d[i]
    b_updated = b_vals + a_plus_one_vals * d_vals
    
    # Store updated b values
    tl.store(b_ptr + offsets, b_updated, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )