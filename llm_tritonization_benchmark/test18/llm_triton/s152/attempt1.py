import triton
import triton.language as tl
import torch

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load d and e
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute b[i] = d[i] * e[i]
    b_vals = d_vals * e_vals
    tl.store(b_ptr + idx, b_vals, mask=mask)
    
    # Load a and c for s152s computation
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # s152s: a[i] += b[i] * c[i]
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + idx, a_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s152_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )