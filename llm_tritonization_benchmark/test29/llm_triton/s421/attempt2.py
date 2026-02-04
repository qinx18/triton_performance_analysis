import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n):
    pid = tl.program_id(0)
    BLOCK_SIZE = 256
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n - 1)
    
    # Load yy[i+1] and a[i]
    yy_offsets = offsets + 1
    yy_vals = tl.load(yy_ptr + yy_offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    n = a.shape[0]
    
    # Copy xx to yy
    yy.copy_(xx)
    
    # Launch kernel for parallel computation
    grid = (triton.cdiv(n - 1, 256),)
    s421_kernel[grid](xx, yy, a, n)
    
    return xx