import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(a_ptr, xx_ptr, yy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load a[i] and yy[i+1]
    a_vals = tl.load(a_ptr + idx, mask=mask)
    yy_vals = tl.load(yy_ptr + idx + 1, mask=mask & ((idx + 1) < (n_elements + 1)))
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + idx, result, mask=mask)

def s421_triton(a, xx, yy):
    # Copy xx to yy first (yy = xx)
    yy.copy_(xx)
    
    n_elements = len(a) - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](a, xx, yy, n_elements, BLOCK_SIZE=BLOCK_SIZE)