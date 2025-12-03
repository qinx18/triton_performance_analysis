import torch
import triton
import triton.language as tl

@triton.jit
def s1251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Compute s = b[i] + c[i]
    s = b_vals + c_vals
    
    # Compute b[i] = a[i] + d[i]
    b_new = a_vals + d_vals
    
    # Compute a[i] = s * e[i]
    a_new = s * e_vals
    
    # Store results
    tl.store(b_ptr + block_start + offsets, b_new, mask=mask)
    tl.store(a_ptr + block_start + offsets, a_new, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1251_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)