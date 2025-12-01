import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(
    flat_2d_array_ptr,
    xx_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    xx_vals = tl.load(xx_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    tl.store(flat_2d_array_ptr + offsets + 1, result, mask=mask)

def s423_triton(flat_2d_array, a):
    vl = 64
    n_elements = flat_2d_array.shape[0] - vl - 1
    
    # xx points to flat_2d_array + vl
    xx = flat_2d_array[vl:]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array,
        xx,
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return flat_2d_array