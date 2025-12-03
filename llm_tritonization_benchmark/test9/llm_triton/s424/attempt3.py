import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(
    a_ptr,
    flat_2d_array_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load from a and flat_2d_array at indices i
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    flat_vals = tl.load(flat_2d_array_ptr + indices, mask=mask, other=0.0)
    
    # Compute
    result = flat_vals + a_vals
    
    # Store to xx at indices i+1
    store_indices = indices + 1
    tl.store(xx_ptr + store_indices, result, mask=mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1
    
    # Create xx array with vl offset (vl = 63)
    vl = 63
    xx_size = a.shape[0] + vl
    xx = torch.zeros(xx_size, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        a,
        flat_2d_array,
        xx,
        n_elements,
        BLOCK_SIZE,
    )
    
    return xx