import triton
import triton.language as tl
import torch

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input arrays
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    # Simplified: b[i] * (c[i] + 2*d[i]) + c[i]*c[i] + d[i]*c[i]
    result = c * b + d * b + c * c + d * b + d * c
    
    # Store to a[2*i] positions
    output_offsets = 2 * offsets
    output_mask = output_offsets < (2 * n_elements)
    tl.store(a_ptr + output_offsets, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    n_elements = len(b) // 2
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1111_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)