import torch
import triton
import triton.language as tl

@triton.jit
def s1421_kernel(a_ptr, b_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    xx_vals = tl.load(xx_ptr + indices, mask=mask)
    
    result = xx_vals + a_vals
    
    tl.store(b_ptr + indices, result, mask=mask)

def s1421_triton(a, b, xx):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1421_kernel[grid](
        a, b, xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )