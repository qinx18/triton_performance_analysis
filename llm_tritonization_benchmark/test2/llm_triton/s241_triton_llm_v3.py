import torch
import triton
import triton.language as tl

@triton.jit
def s241_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current elements
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # First statement: a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Load a[i+1] for second statement
    offsets_plus1 = offsets + 1
    mask_plus1 = offsets_plus1 < (n_elements + 1)  # Allow loading a[i+1] where i+1 can equal n_elements
    a_vals_plus1 = tl.load(a_ptr + offsets_plus1, mask=mask_plus1)
    
    # Second statement: b[i] = a[i] * a[i+1] * d[i]
    # Only compute where we have valid a[i+1]
    valid_mask = mask & (offsets + 1 < (n_elements + 1))
    b_new_vals = a_vals * a_vals_plus1 * d_vals
    tl.store(b_ptr + offsets, b_new_vals, mask=valid_mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )