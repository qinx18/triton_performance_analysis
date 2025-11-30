import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute updates
    a_new = b_vals + c_vals  # for a[i+1]
    b_new = c_vals * e_vals  # for b[i]
    d_new = a_vals * e_vals  # for d[i]
    
    # Store results
    # a[i+1] = b[i] + c[i] - store at offset+1
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)
    tl.store(a_ptr + offsets_plus_1, a_new, mask=mask_plus_1)
    
    # b[i] and d[i] - store at original offset
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(d_ptr + offsets, d_new, mask=mask)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )