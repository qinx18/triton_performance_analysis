import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load yy values (yy[i+1])
    yy_offsets = block_start + offsets + 1
    yy_mask = yy_offsets < n_elements
    yy_vals = tl.load(yy_ptr + yy_offsets, mask=yy_mask)
    
    # Load a values (a[i])
    a_offsets = block_start + offsets
    a_mask = a_offsets < (n_elements - 1)
    a_vals = tl.load(a_ptr + a_offsets, mask=a_mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + a_offsets, result, mask=a_mask)

def s421_triton(a, xx, yy):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Copy xx to yy (yy = xx)
    yy.copy_(xx)
    
    s421_kernel[grid](
        xx, yy, a, n_elements, BLOCK_SIZE
    )