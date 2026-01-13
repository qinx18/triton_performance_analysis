import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load yy[i+1] (shifted read)
    yy_indices = indices + 1
    yy_mask = yy_indices < (n_elements + 1)  # Allow reading up to n_elements
    yy_vals = tl.load(yy_ptr + yy_indices, mask=yy_mask, other=0.0)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + indices, result, mask=mask)

def s421_triton(a, xx, yy):
    N = a.shape[0] - 1  # Loop goes to LEN_1D - 1
    
    # Copy xx to yy (equivalent to yy = xx)
    yy.copy_(xx)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s421_kernel[grid](
        xx, yy, a, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx