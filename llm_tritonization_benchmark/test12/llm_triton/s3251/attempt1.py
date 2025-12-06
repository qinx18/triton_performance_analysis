import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load current values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute updates
    new_b = c_vals * e_vals
    new_d = a_vals * e_vals
    new_a = b_vals + c_vals
    
    # Store results
    tl.store(b_ptr + idx, new_b, mask=mask)
    tl.store(d_ptr + idx, new_d, mask=mask)
    
    # Store a[i+1] = b[i] + c[i]
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < (n_elements + 1)
    tl.store(a_ptr + idx_plus_1, new_a, mask=mask_plus_1)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )