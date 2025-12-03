import torch
import triton
import triton.language as tl

@triton.jit
def s422_kernel(
    flat_2d_array_ptr,
    a_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load from flat_2d_array[i + 8] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + indices + 8, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute xx[i] = flat_2d_array[i + 8] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + indices, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s422_kernel[grid](
        flat_2d_array,
        a,
        xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )