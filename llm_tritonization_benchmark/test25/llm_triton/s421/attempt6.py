import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < (n_elements - 1)
    
    # Load yy[i+1] + a[i]
    yy_vals = tl.load(yy_ptr + idx + 1, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    result = yy_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + idx, result, mask=mask)

def s421_triton(a, xx, yy):
    N = a.shape[0]
    
    # Copy xx to yy (yy = xx)
    yy.copy_(xx)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s421_kernel[grid](xx, yy, a, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return xx