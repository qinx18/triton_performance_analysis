import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    result = flat_vals + a_vals
    
    tl.store(flat_2d_array_ptr + idx + 64, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    STRIP_SIZE = 64
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_strip_size = strip_end - strip_start
        
        if current_strip_size > 0:
            s424_kernel[(1,)](
                flat_2d_array,
                a,
                strip_start,
                strip_end,
                BLOCK_SIZE=current_strip_size
            )