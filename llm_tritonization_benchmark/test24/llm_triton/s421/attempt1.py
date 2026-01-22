import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute indices for xx[i] = yy[i+1] + a[i]
    xx_indices = block_start + offsets
    yy_indices = xx_indices + 1  # yy[i+1]
    
    # Create masks
    xx_mask = xx_indices < (n - 1)  # Process up to n-2 (since loop is i < n-1)
    yy_mask = yy_indices < n        # yy[i+1] must be valid
    
    # Load data
    yy_vals = tl.load(yy_ptr + yy_indices, mask=yy_mask, other=0.0)
    a_vals = tl.load(a_ptr + xx_indices, mask=xx_mask, other=0.0)
    
    # Compute
    result = yy_vals + a_vals
    
    # Store
    tl.store(xx_ptr + xx_indices, result, mask=xx_mask)

def s421_triton(a, xx, yy):
    n = a.shape[0]
    
    # Copy xx to yy (equivalent to yy = xx)
    yy.copy_(xx)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s421_kernel[grid](xx, yy, a, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return xx