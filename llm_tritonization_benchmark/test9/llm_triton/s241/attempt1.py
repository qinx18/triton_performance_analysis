import torch
import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load initial values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # First computation: a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    
    # Store a[i]
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # For second computation, we need a[i+1]
    # Load a[i+1] with shifted indices
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < (n_elements + 1)  # Allow reading a[i+1] for valid indices
    a_vals_plus1 = tl.load(a_ptr + idx_plus1, mask=mask_plus1, other=0.0)
    
    # Second computation: b[i] = a[i] * a[i+1] * d[i]
    b_new_vals = a_vals * a_vals_plus1 * d_vals
    
    # Store new b[i]
    tl.store(b_ptr + idx, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop runs from 0 to LEN_1D-2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )