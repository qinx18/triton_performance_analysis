import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(
    xx_ptr,
    yy_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load yy[i+1] values (shifted by 1)
    yy_offsets = offsets + 1
    yy_mask = yy_offsets < (n_elements + 1)
    yy_vals = tl.load(yy_ptr + yy_offsets, mask=yy_mask, other=0.0)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store back to xx[i]
    tl.store(xx_ptr + offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Copy xx to yy (yy = xx)
    yy[:].copy_(xx[:])
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](
        xx,
        yy,
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx