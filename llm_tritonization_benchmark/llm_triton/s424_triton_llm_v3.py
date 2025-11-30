import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    flat_2d_array_vals = tl.load(flat_2d_array_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    result = flat_2d_array_vals + a_vals
    
    tl.store(xx_ptr + offsets + 1, result, mask=mask)

def s424_triton(flat_2d_array, a):
    n_elements = a.shape[0] - 1
    
    # Create xx array (offset by 63 elements as in original)
    vl = 63
    xx = torch.zeros_like(flat_2d_array)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array, xx[vl:], a, n_elements, BLOCK_SIZE
    )
    
    return xx