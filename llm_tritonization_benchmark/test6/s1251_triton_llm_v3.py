import triton
import triton.language as tl
import torch

@triton.jit
def s1251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    b_val = tl.load(b_ptr + idx, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    a_val = tl.load(a_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # Compute
    s = b_val + c_val
    b_new = a_val + d_val
    a_new = s * e_val
    
    # Store results
    tl.store(b_ptr + idx, b_new, mask=mask)
    tl.store(a_ptr + idx, a_new, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1251_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )
    
    return a, b