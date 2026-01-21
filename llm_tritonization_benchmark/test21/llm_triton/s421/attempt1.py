import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n - 1)
    
    # Load yy[i+1] values
    yy_offsets = offsets + 1
    yy_vals = tl.load(yy_ptr + yy_offsets, mask=mask)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    n = a.shape[0]
    
    # Copy yy to xx first (yy = xx equivalent)
    xx.copy_(yy)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s421_kernel[grid](xx, xx, a, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return xx