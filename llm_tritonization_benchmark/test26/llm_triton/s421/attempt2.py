import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(
    xx_ptr, yy_ptr, a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load yy[i+1] and a[i] for i in [0, n_elements-1]
    yy_indices = indices + 1
    
    yy_vals = tl.load(yy_ptr + yy_indices, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + indices, result, mask=mask)

def s421_triton(a, xx, yy):
    n_elements = a.shape[0] - 1
    
    # yy = xx (copy xx to yy first)
    yy.copy_(xx)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](
        xx, yy, a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx