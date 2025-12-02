import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    flat_vals = tl.load(flat_2d_array_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    result = flat_vals + a_vals
    
    tl.store(xx_ptr + indices + 1, result, mask=mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1
    
    # Create xx array with offset (equivalent to xx = flat_2d_array + 63)
    vl = 63
    xx = torch.zeros_like(flat_2d_array)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array, xx, a,
        n_elements, BLOCK_SIZE
    )
    
    # Apply the offset by copying from position vl onwards
    result = torch.zeros_like(flat_2d_array)
    result[vl:] = xx[:-vl] if vl > 0 else xx
    
    return result