import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_len, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Read from flat_2d_array[idx] and a[idx]
    read_mask = offsets < strip_len
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=read_mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=read_mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[idx+1] = flat_2d_array[idx+64]
    write_idx = idx + 64
    tl.store(flat_2d_array_ptr + write_idx, result, mask=read_mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in strips of 64 elements sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_len = min(STRIP_SIZE, n_elements - strip_start)
        
        if strip_len > 0:
            grid = (1,)
            s424_kernel[grid](
                flat_2d_array,
                a,
                strip_start,
                strip_len,
                BLOCK_SIZE=BLOCK_SIZE
            )