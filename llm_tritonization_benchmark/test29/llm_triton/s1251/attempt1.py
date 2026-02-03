import triton
import triton.language as tl
import torch

@triton.jit
def s1251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    b_val = tl.load(b_ptr + indices, mask=mask)
    c_val = tl.load(c_ptr + indices, mask=mask)
    a_val = tl.load(a_ptr + indices, mask=mask)
    d_val = tl.load(d_ptr + indices, mask=mask)
    e_val = tl.load(e_ptr + indices, mask=mask)
    
    # Scalar expansion: s = b[i] + c[i]
    s = b_val + c_val
    
    # b[i] = a[i] + d[i]
    b_new = a_val + d_val
    
    # a[i] = s * e[i]
    a_new = s * e_val
    
    # Store results
    tl.store(b_ptr + indices, b_new, mask=mask)
    tl.store(a_ptr + indices, a_new, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )