import triton
import triton.language as tl
import torch

@triton.jit
def s1281_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute x = b[i]*c[i] + a[i]*d[i] + e[i]
    x_vals = b_vals * c_vals + a_vals * d_vals + e_vals
    
    # Store results
    # a[i] = x - 1.0
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    # b[i] = x
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1281_kernel[grid](
        a, b, c, d, e, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )