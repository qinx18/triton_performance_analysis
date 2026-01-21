import torch
import triton
import triton.language as tl

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d and e arrays
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute b[i] = d[i] * e[i]
    b_vals = d_vals * e_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Load b and c for the s152s operation
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Perform s152s: a[i] += b[i] * c[i]
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s152_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )