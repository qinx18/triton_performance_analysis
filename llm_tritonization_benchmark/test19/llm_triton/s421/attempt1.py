import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load yy[i+1] (shifted read)
    yy_indices = indices + 1
    yy_vals = tl.load(yy_ptr + yy_indices, mask=mask)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + indices, result, mask=mask)

def s421_triton(a, xx, yy):
    # Copy xx to yy (yy = xx)
    yy.copy_(xx)
    
    n_elements = a.shape[0] - 1  # Process LEN_1D - 1 elements
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](
        xx, yy, a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx