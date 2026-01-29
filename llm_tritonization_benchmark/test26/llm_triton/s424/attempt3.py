import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array, a, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < strip_start + strip_size
    
    vals_flat = tl.load(flat_2d_array + idx, mask=mask)
    vals_a = tl.load(a + idx, mask=mask)
    
    result = vals_flat + vals_a
    
    write_idx = idx + 64
    tl.store(flat_2d_array + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        current_strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        if current_strip_size > 0:
            s424_kernel[(1,)](
                flat_2d_array,
                a,
                strip_start,
                current_strip_size,
                BLOCK_SIZE=BLOCK_SIZE
            )