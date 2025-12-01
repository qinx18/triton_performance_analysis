import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load xx[i] and a[i]
    xx_vals = tl.load(xx_ptr + i_offsets, mask=mask)
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Compute xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    tl.store(flat_2d_array_ptr + i_offsets + 1, result, mask=mask)

def s423_triton(flat_2d_array, a):
    vl = 64
    n_elements = flat_2d_array.shape[0] - vl - 1
    
    # xx points to flat_2d_array + vl
    xx_ptr = flat_2d_array.data_ptr() + vl * flat_2d_array.element_size()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array,
        xx_ptr,
        a,
        n_elements,
        BLOCK_SIZE
    )
    
    return flat_2d_array