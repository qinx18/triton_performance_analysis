import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel(
    xx_ptr, yy_ptr, a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load yy[i+1] and a[i]
    yy_offsets = offsets + 1
    yy_mask = yy_offsets < (n_elements + 1)
    yy_vals = tl.load(yy_ptr + yy_offsets, mask=yy_mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute xx[i] = yy[i+1] + a[i]
    result = yy_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + offsets, result, mask=mask)

def s421_triton(flat_2d_array, a):
    n_elements = len(a) - 1  # LEN_1D - 1
    
    # Create copy for yy (read-only copy to handle WAR dependency)
    xx = flat_2d_array.clone()
    yy = xx.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s421_kernel[grid](
        xx, yy, a,
        n_elements,
        BLOCK_SIZE,
    )
    
    # Copy result back to original array
    flat_2d_array[:n_elements] = xx[:n_elements]
    
    return flat_2d_array