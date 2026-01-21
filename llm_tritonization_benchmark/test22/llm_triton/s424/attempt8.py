import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = offsets < strip_size
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    result = flat_vals + a_vals
    
    write_idx = idx + 64
    tl.store(flat_2d_array_ptr + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array,
            a,
            strip_start,
            strip_size,
            BLOCK_SIZE=BLOCK_SIZE
        )