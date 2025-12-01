import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    flat_vals = tl.load(flat_2d_array_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    result = flat_vals + a_vals
    
    tl.store(xx_ptr + indices + 1, result, mask=mask)

def s424_triton(flat_2d_array, a):
    LEN_1D = flat_2d_array.size(0)
    n_elements = LEN_1D - 1
    
    vl = 63
    xx = torch.zeros(LEN_1D + vl, dtype=flat_2d_array.dtype, device=flat_2d_array.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array, a, xx, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx