import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    yy_vals = tl.load(yy_ptr + offsets + 1, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    result = yy_vals + a_vals
    
    tl.store(xx_ptr + offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](xx, yy, a, n_elements, BLOCK_SIZE)
    
    return xx