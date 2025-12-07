import triton
import triton.language as tl
import torch

@triton.jit
def s1251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute: s = b[i] + c[i]
    s = b_vals + c_vals
    
    # Compute: b[i] = a[i] + d[i]
    new_b = a_vals + d_vals
    
    # Compute: a[i] = s * e[i]
    new_a = s * e_vals
    
    # Store results
    tl.store(b_ptr + idx, new_b, mask=mask)
    tl.store(a_ptr + idx, new_a, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )