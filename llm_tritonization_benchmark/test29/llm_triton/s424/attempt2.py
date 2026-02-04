import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array, a, n_elements, vl, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for strip_start in range(0, n_elements, STRIP_SIZE):
        current_offsets = strip_start + offsets
        mask = current_offsets < n_elements
        
        flat_vals = tl.load(flat_2d_array + current_offsets, mask=mask)
        a_vals = tl.load(a + current_offsets, mask=mask)
        
        result = flat_vals + a_vals
        
        write_offsets = current_offsets + vl + 1
        tl.store(flat_2d_array + write_offsets, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    n_elements = N - 1
    vl = 63
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    s424_kernel[(1,)](flat_2d_array, a, n_elements, vl, STRIP_SIZE, BLOCK_SIZE)