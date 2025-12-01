import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program id and compute offsets once
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # First pass: compute a[i] = b[i] * c[i] * d[i]
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # Second pass: compute b[i] = a[i] * a[i+1] * d[i]
    # Load a[i] values
    a_i_vals = tl.load(a_ptr + current_offsets, mask=mask)
    
    # Load a[i+1] values
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n_elements + 1)  # Allow reading one past for a[i+1]
    a_i_plus_1_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
    
    # Compute b[i] = a[i] * a[i+1] * d[i]
    b_new_vals = a_i_vals * a_i_plus_1_vals * d_vals
    tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop runs from 0 to LEN_1D-2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b