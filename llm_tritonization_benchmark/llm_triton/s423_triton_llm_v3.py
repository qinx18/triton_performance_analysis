import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[i] and xx[i] (which is flat_2d_array[i + 64])
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    xx_vals = tl.load(flat_2d_array_ptr + offsets + 64, mask=mask)
    
    # Compute flat_2d_array[i+1] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    tl.store(flat_2d_array_ptr + offsets + 1, result, mask=mask)

def s423_triton(flat_2d_array, a):
    n_elements = len(a) - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, a, n_elements, BLOCK_SIZE
    )
    
    return flat_2d_array