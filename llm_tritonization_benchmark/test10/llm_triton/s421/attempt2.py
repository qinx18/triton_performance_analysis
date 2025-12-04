import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(a_ptr, xx_ptr, yy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load a[i] and yy[i+1]
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # For yy[i+1], we need to check bounds for i+1 < total_size
    yy_idx = idx + 1
    yy_mask = (idx < n_elements) & (yy_idx < (n_elements + 1))
    yy_vals = tl.load(yy_ptr + yy_idx, mask=yy_mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + idx, result, mask=mask)

def s421_triton(a, xx, yy):
    # yy = xx (copy xx to yy first)
    yy.copy_(xx)
    
    # Process LEN_1D - 1 elements
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](
        a, xx, yy, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx