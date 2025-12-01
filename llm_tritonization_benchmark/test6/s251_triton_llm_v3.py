import torch
import triton
import triton.language as tl

@triton.jit
def s251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute: s = b[i] + c[i] * d[i]; a[i] = s * s
    s = b_vals + c_vals * d_vals
    result = s * s
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s251_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a