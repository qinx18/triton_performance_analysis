import torch
import triton
import triton.language as tl

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, t, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    e_vals = tl.load(e_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Apply conditional update
    condition = e_vals >= t
    update_mask = mask & condition
    
    # Update a and b where condition is true
    a_new = a_vals + c_vals * d_vals
    b_new = b_vals + c_vals * c_vals
    
    # Store results
    tl.store(a_ptr + indices, a_new, mask=update_mask)
    tl.store(b_ptr + indices, b_new, mask=update_mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e,
        n_elements, t,
        BLOCK_SIZE=BLOCK_SIZE
    )