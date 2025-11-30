import torch
import triton
import triton.language as tl

@triton.jit
def s243_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Load a[i+1] values from copy
    offsets_plus1 = offsets + 1
    mask_plus1 = offsets_plus1 < (n_elements + 1)  # Allow reading a[i+1] for valid range
    a_plus1_vals = tl.load(a_copy_ptr + offsets_plus1, mask=mask_plus1, other=0.0)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_new_vals = a_vals + d_vals * e_vals
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_final_vals = b_new_vals + a_plus1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_final_vals, mask=mask)
    tl.store(b_ptr + offsets, b_new_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = len(a) - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy of array a for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e,
        n_elements,
        BLOCK_SIZE,
    )