import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b[i], c[i], d[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    result = c_vals * b_vals + d_vals * b_vals + c_vals * c_vals + d_vals * b_vals + d_vals * c_vals
    
    # Store to a[2*i]
    output_indices = 2 * indices
    output_mask = mask & (output_indices < 2 * n_elements)
    tl.store(a_ptr + output_indices, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1111_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a