import triton
import triton.language as tl
import torch

@triton.jit
def s1421_kernel(
    a_ptr, b_ptr, xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    xx_vals = tl.load(xx_ptr + idx, mask=mask)
    
    result = xx_vals + a_vals
    
    tl.store(b_ptr + idx, result, mask=mask)

def s1421_triton(a, b, xx):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1421_kernel[grid](
        a, b, xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )