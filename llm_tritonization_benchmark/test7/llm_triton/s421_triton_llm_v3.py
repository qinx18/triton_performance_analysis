import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Check if we're in the valid range (i < n_elements - 1)
    indices = block_start + offsets
    mask = indices < (n_elements - 1)
    
    # Load yy[i+1] and a[i]
    yy_vals = tl.load(yy_ptr + indices + 1, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + indices, result, mask=mask)

def s421_triton(a, xx, yy):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Copy yy to xx first (yy = xx becomes xx = yy for the computation)
    xx.copy_(yy)
    
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    s421_kernel[grid](
        xx, yy, a, n_elements, BLOCK_SIZE
    )