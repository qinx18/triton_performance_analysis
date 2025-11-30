import triton
import triton.language as tl
import torch

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
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from flat_2d_array[i + 8] and a[i]
    flat_2d_offsets = offsets + 8
    flat_2d_vals = tl.load(flat_2d_array_ptr + flat_2d_offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute xx[i] = flat_2d_array[i + 8] + a[i]
    result = flat_2d_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + offsets, result, mask=mask)

def s422_triton(flat_2d_array, a):
    n_elements = a.shape[0]
    
    # Create xx as a view into flat_2d_array starting at offset 4
    xx = flat_2d_array[4:4+n_elements]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s422_kernel[grid](
        xx,
        flat_2d_array,
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx