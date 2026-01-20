import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
    vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    result = vals + a_vals
    
    write_idx = idx + 64
    write_mask = write_idx < flat_2d_array_ptr.shape[0]
    final_mask = mask & write_mask
    
    tl.store(flat_2d_array_ptr + write_idx, result, mask=final_mask)

def s424_triton(a, flat_2d_array, xx):
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    n_elements = a.shape[0] - 1
    
    # Process in sequential strips due to RAW dependency
    for strip_start in range(0, n_elements, STRIP_SIZE):
        current_strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        if current_strip_size <= 0:
            break
            
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array, a, strip_start, n_elements,
            BLOCK_SIZE=min(BLOCK_SIZE, current_strip_size)
        )