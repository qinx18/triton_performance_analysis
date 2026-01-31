import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(a_ptr, flat_2d_array_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load a[i] and xx[i]
    a_vals = tl.load(a_ptr + idx, mask=mask)
    xx_vals = tl.load(xx_ptr + idx, mask=mask)
    
    # Compute flat_2d_array[i+1] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    tl.store(flat_2d_array_ptr + (idx + 1), result, mask=mask)

def s423_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        a, flat_2d_array, xx,
        n_elements, BLOCK_SIZE
    )
    
    return flat_2d_array