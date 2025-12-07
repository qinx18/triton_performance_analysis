import torch
import triton
import triton.language as tl

@triton.jit
def s251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Compute: s = b[i] + c[i] * d[i]
    s = b_vals + c_vals * d_vals
    
    # Compute: a[i] = s * s
    a_vals = s * s
    
    # Store result
    tl.store(a_ptr + idx, a_vals, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s251_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)