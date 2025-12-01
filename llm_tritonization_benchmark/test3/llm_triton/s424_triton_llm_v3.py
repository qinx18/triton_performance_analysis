import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(
    flat_2d_array_ptr,
    a_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    flat_2d_vals = tl.load(flat_2d_array_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute xx[i+1] = flat_2d_array[i] + a[i]
    result = flat_2d_vals + a_vals
    
    # Store to xx[i+1] (offset by 1)
    tl.store(xx_ptr + offsets + 1, result, mask=mask)

def s424_triton(flat_2d_array, a):
    n_elements = flat_2d_array.shape[0] - 1
    vl = 63
    
    # Create xx array (flat_2d_array + offset of vl)
    xx = torch.zeros_like(flat_2d_array)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array,
        a,
        xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx