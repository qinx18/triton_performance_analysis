import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load arrays for first computation
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # First computation: a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # For second computation, we need a[i+1] values
    # Load a[i+1] with offset
    indices_plus1 = indices + 1
    mask_plus1 = indices_plus1 < (n_elements + 1)
    a_vals_plus1 = tl.load(a_ptr + indices_plus1, mask=mask_plus1)
    
    # Second computation: b[i] = a[i] * a[i+1] * d[i]
    # Only compute for valid indices (where i+1 is valid)
    b_new_vals = a_vals * a_vals_plus1 * d_vals
    tl.store(b_ptr + indices, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b