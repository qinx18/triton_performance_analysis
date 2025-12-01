import triton
import triton.language as tl
import torch

@triton.jit
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all input values
    a_val = tl.load(a_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr + offsets, mask=mask)
    c_val = tl.load(c_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    e_val = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute x = b[i]*c[i] + a[i]*d[i] + e[i]
    x = b_val * c_val + a_val * d_val + e_val
    
    # Update arrays
    a_new = x - 1.0
    b_new = x
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s1281_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1281_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )