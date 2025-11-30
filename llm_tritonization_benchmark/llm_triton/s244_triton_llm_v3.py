import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load initial values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Second statement: b[i] = c[i] + b[i]
    b_vals = c_vals + b_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Third statement: a[i+1] = b[i] + a[i+1] * d[i]
    # Need to handle the shift carefully
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)
    
    # Load a[i+1] values
    a_plus_1_vals = tl.load(a_ptr + offsets_plus_1, mask=mask_plus_1)
    
    # Compute new a[i+1] values
    new_a_plus_1 = b_vals + a_plus_1_vals * d_vals
    
    # Store back to a[i+1]
    tl.store(a_ptr + offsets_plus_1, new_a_plus_1, mask=mask_plus_1)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )