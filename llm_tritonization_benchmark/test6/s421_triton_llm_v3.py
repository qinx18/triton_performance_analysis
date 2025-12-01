import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load yy values (xx[i+1])
    yy_offsets = block_start + offsets + 1
    yy_mask = yy_offsets < (n_elements + 1)
    yy_vals = tl.load(xx_ptr + yy_offsets, mask=yy_mask, other=0.0)
    
    # Load a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + block_start + offsets, result, mask=mask)

def s421_triton(xx, a):
    n_elements = len(a) - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create a copy for yy (read-only copy to handle WAR dependency)
    yy = xx.clone()
    
    s421_kernel[grid](
        yy, a, n_elements, BLOCK_SIZE
    )
    
    return xx