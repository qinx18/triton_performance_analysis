import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute updates
    new_a_vals = b_vals + c_vals  # for a[i+1]
    new_b_vals = c_vals * e_vals  # for b[i]
    new_d_vals = a_vals * e_vals  # for d[i]
    
    # Store updates
    # a[i+1] = b[i] + c[i] (shift indices by +1)
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < (n_elements + 1)  # Allow one extra element
    tl.store(a_ptr + idx_plus1, new_a_vals, mask=mask_plus1)
    
    # b[i] = c[i] * e[i]
    tl.store(b_ptr + idx, new_b_vals, mask=mask)
    
    # d[i] = a[i] * e[i]
    tl.store(d_ptr + idx, new_d_vals, mask=mask)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )