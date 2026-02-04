import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array, a, n_elements, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    flat_vals = tl.load(flat_2d_array + offsets, mask=mask)
    a_vals = tl.load(a + offsets, mask=mask)
    
    result = flat_vals + a_vals
    
    write_offsets = offsets + 64
    write_mask = write_offsets < (n_elements + 64)
    tl.store(flat_2d_array + write_offsets, result, mask=mask & write_mask)

def s424_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    n_elements = N - 1
    
    BLOCK_SIZE = 64
    grid = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s424_kernel[(grid,)](flat_2d_array, a, n_elements, BLOCK_SIZE, BLOCK_SIZE)