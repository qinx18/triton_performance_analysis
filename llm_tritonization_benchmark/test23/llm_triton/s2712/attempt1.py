import torch
import triton
import triton.language as tl

@triton.jit
def s2712_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Conditional update: if a[i] > b[i]: a[i] += b[i] * c[i]
    condition = a_vals > b_vals
    updated_a = tl.where(condition, a_vals + b_vals * c_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + indices, updated_a, mask=mask)

def s2712_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2712_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )