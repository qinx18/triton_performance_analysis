import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load yy[i+1] (shifted by 1) and a[i]
        yy_vals = tl.load(yy_ptr + current_offsets + 1, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute xx[i] = yy[i+1] + a[i]
        result = yy_vals + a_vals
        
        # Store back to xx
        tl.store(xx_ptr + current_offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    n = a.shape[0] - 1  # Loop runs from 0 to LEN_1D - 1
    BLOCK_SIZE = 256
    
    # Copy xx to yy before kernel execution
    yy.copy_(xx)
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s421_kernel[grid](xx, yy, a, n, BLOCK_SIZE=BLOCK_SIZE)