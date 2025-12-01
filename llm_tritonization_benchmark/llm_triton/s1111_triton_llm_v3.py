import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    current_offsets = block_start + offsets
    
    # Load b[i], c[i], d[i]
    b = tl.load(b_ptr + current_offsets, mask=mask)
    c = tl.load(c_ptr + current_offsets, mask=mask)
    d = tl.load(d_ptr + current_offsets, mask=mask)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    result = c * b + d * b + c * c + d * b + d * c
    
    # Store to a[2*i]
    output_offsets = 2 * current_offsets
    output_mask = mask & (output_offsets < 2 * n_elements)
    tl.store(a_ptr + output_offsets, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1111_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )