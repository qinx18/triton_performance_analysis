import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(a_ptr, xx_ptr, yy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block of indices to process
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load yy[i+1] values (shift indices by 1)
    yy_offsets = offsets + 1
    yy_mask = mask & (yy_offsets < (n_elements + 1))
    yy_vals = tl.load(yy_ptr + yy_offsets, mask=yy_mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result to xx[i]
    tl.store(xx_ptr + offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    # First copy xx to yy (yy = xx)
    yy.copy_(xx)
    
    # Process LEN_1D - 1 elements
    n_elements = a.shape[0] - 1
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s421_kernel[grid](
        a, xx, yy, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx