import triton
import triton.language as tl
import torch

@triton.jit
def s2712_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Conditional computation: if a[i] > b[i]: a[i] += b[i] * c[i]
    condition = a_vals > b_vals
    update = b_vals * c_vals
    result = tl.where(condition, a_vals + update, a_vals)
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s2712_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2712_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
    
    return a