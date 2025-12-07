import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid elements (we process LEN_1D - 1 elements)
    mask = idx < n_elements
    
    # Load yy[i+1] values (which is xx[i+1] since yy = xx)
    yy_offsets = idx + 1
    yy_mask = yy_offsets < (n_elements + 1)  # Allow reading one extra element
    yy_vals = tl.load(xx_ptr + yy_offsets, mask=yy_mask)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store back to xx[i]
    tl.store(xx_ptr + idx, result, mask=mask)

def s421_triton(a, xx, yy):
    # Copy xx to yy (yy = xx)
    yy.copy_(xx)
    
    # We process LEN_1D - 1 elements
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](xx, a, n_elements, BLOCK_SIZE)
    
    return xx