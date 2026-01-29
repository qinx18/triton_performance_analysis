import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    strip_id = tl.program_id(0)
    strip_start = strip_id * 64
    
    idx = strip_start + offsets
    mask = idx < n_elements
    
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    result = flat_vals + a_vals
    
    tl.store(xx_ptr + idx + 1, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    
    BLOCK_SIZE = 64
    num_strips = triton.cdiv(n_elements, BLOCK_SIZE)
    
    for strip_id in range(num_strips):
        s424_kernel[(1,)](flat_2d_array, a, xx, n_elements, BLOCK_SIZE=BLOCK_SIZE)