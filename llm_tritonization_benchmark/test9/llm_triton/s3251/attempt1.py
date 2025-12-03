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
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute operations
    # a[i+1] = b[i] + c[i]
    a_new = b_vals + c_vals
    tl.store(a_ptr + idx + 1, a_new, mask=mask)
    
    # b[i] = c[i] * e[i]
    b_new = c_vals * e_vals
    tl.store(b_ptr + idx, b_new, mask=mask)
    
    # d[i] = a[i] * e[i]
    d_new = a_vals * e_vals
    tl.store(d_ptr + idx, d_new, mask=mask)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )