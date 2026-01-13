import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Condition: c[i] < 0
    condition = c_vals < 0.0
    
    # For condition true (c[i] < 0): b[i] = a[i] + d[i] * d[i]
    b_vals = a_vals + d_vals * d_vals
    tl.store(b_ptr + idx, b_vals, mask=mask & condition)
    
    # For condition false (c[i] >= 0): a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    tl.store(a_ptr + idx, a_new, mask=mask & ~condition)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )