import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n):
    pid = tl.program_id(0)
    BLOCK_SIZE = 256
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    mask = current_offsets < (n - 1)
    
    # Load yy[i+1] and a[i]
    yy_vals = tl.load(yy_ptr + current_offsets + 1, mask=mask)
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + current_offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    n = a.shape[0]
    
    # Copy xx to yy
    yy.copy_(xx)
    
    # Launch kernel
    grid = (triton.cdiv(n - 1, 256),)
    s421_kernel[grid](xx, yy, a, n)
    
    return xx