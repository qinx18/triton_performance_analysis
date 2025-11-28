import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load yy[i+1] and a[i]
    yy_vals = tl.load(yy_ptr + offsets + 1, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + offsets, result, mask=mask)

def s421_triton(xx, a):
    n_elements = xx.shape[0] - 1  # LEN_1D - 1
    
    # Create yy as a copy of xx (equivalent to yy = xx)
    yy = xx.clone()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](
        xx, yy, a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx