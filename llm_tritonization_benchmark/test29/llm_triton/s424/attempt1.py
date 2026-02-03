import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for strip_start in range(0, n_elements, STRIP_SIZE):
        idx = strip_start + offsets
        mask = idx < n_elements
        
        flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
        a_vals = tl.load(a_ptr + idx, mask=mask)
        
        result = flat_vals + a_vals
        
        write_mask = (idx + 64) < (n_elements + 64)
        write_mask = write_mask & mask
        tl.store(flat_2d_array_ptr + idx + 64, result, mask=write_mask)

def s424_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    n_elements = N - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    s424_kernel[(1,)](
        flat_2d_array, 
        a, 
        n_elements, 
        STRIP_SIZE=STRIP_SIZE, 
        BLOCK_SIZE=BLOCK_SIZE
    )