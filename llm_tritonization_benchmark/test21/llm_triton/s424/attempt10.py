import triton
import triton.language as tl
import torch

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
    STRIP_SIZE = 64
    n_elements = a.shape[0] - 1
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        current_strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        if current_strip_size > 0:
            block_size = 1
            while block_size < current_strip_size:
                block_size *= 2
            
            s424_kernel[(1,)](
                flat_2d_array,
                a,
                strip_start,
                current_strip_size,
                BLOCK_SIZE=block_size
            )