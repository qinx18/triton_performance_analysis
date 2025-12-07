import torch
import triton
import triton.language as tl

@triton.jit
def s422_kernel(
    xx_ptr,
    flat_2d_array_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load flat_2d_array[i + 8] + a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx + 8, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    result = flat_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + idx, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s422_kernel[grid](
        xx,
        flat_2d_array,
        a,
        n_elements,
        BLOCK_SIZE,
    )
    
    return xx