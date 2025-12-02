import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load current values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute updates
    new_a_vals = b_vals + c_vals
    new_b_vals = c_vals * e_vals
    new_d_vals = a_vals * e_vals
    
    # Store results
    # a[i+1] = b[i] + c[i] - need to handle offset by 1
    a_indices_plus_1 = indices + 1
    a_mask_plus_1 = a_indices_plus_1 < (n_elements + 1)  # a has size n_elements+1
    tl.store(a_ptr + a_indices_plus_1, new_a_vals, mask=mask & a_mask_plus_1)
    
    # b[i] = c[i] * e[i]
    tl.store(b_ptr + indices, new_b_vals, mask=mask)
    
    # d[i] = a[i] * e[i]
    tl.store(d_ptr + indices, new_d_vals, mask=mask)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )