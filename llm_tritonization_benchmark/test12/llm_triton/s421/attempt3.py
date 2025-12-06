import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, yy_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load yy[i+1] and a[i]
    yy_offsets = current_offsets + 1
    yy_vals = tl.load(yy_ptr + yy_offsets, mask=mask)
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + current_offsets, result, mask=mask)

def s421_triton(a, xx, yy):
    # First copy xx to yy (yy = xx)
    yy.copy_(xx)
    
    n_elements = a.shape[0] - 1  # Process LEN_1D - 1 elements
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](xx, yy, a, n_elements, BLOCK_SIZE)