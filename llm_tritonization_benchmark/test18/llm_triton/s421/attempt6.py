import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n):
    pid = tl.program_id(0)
    BLOCK_SIZE = 256

    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load yy[i+1] + a[i] for xx[i] = yy[i+1] + a[i]
        yy_vals = tl.load(yy_ptr + current_offsets + 1, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        result = yy_vals + a_vals
        
        tl.store(xx_ptr + current_offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    n = a.shape[0]
    
    # Copy xx to yy (yy = xx)
    yy.copy_(xx)
    
    grid = (1,)
    s421_kernel[grid](xx, yy, a, n)