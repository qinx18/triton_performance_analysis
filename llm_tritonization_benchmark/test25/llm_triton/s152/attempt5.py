import triton
import triton.language as tl
import torch

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # b[i] = d[i] * e[i]
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    b_vals = d_vals * e_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # s152s: a[i] += b[i] * c[i]
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = d.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s152_kernel[grid](
        a.data_ptr(), b.data_ptr(), c.data_ptr(), d.data_ptr(), e.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )