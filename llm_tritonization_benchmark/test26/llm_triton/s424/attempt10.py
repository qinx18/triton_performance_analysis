import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, vl, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    mask = offsets < strip_size
    
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    result = flat_vals + a_vals
    
    tl.store(flat_2d_array_ptr + idx + vl + 1, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    LEN_1D = a.shape[0]
    vl = 63
    
    # xx points to flat_2d_array + vl, so initialize it
    xx[:] = flat_2d_array[vl:vl + xx.shape[0]]
    
    n_elements = LEN_1D - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        s424_kernel[(1,)](flat_2d_array, a, strip_start, strip_size, vl, BLOCK_SIZE=BLOCK_SIZE)
    
    # Update xx to reflect changes
    xx[:] = flat_2d_array[vl:vl + xx.shape[0]]