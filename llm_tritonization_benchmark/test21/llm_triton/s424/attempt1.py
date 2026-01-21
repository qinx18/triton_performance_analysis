import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Read from flat_2d_array[i] and a[i]
    read_mask = offsets < n_elements
    flat_vals = tl.load(flat_2d_array_ptr + offsets, mask=read_mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=read_mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to flat_2d_array[i+64] (xx[i+1] = flat_2d_array + 63 + (i+1))
    write_offsets = offsets + 64
    write_mask = offsets < n_elements
    tl.store(flat_2d_array_ptr + write_offsets, result, mask=write_mask)

def s424_triton(a, flat_2d_array, xx):
    # xx = flat_2d_array + 63, so xx[i+1] = flat_2d_array[i+64]
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in strips of 64 elements sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        strip_len = strip_end - strip_start
        
        if strip_len > 0:
            # Launch kernel for this strip
            grid = (1,)
            s424_kernel[grid](
                flat_2d_array[strip_start:],
                a[strip_start:],
                strip_len,
                BLOCK_SIZE=BLOCK_SIZE
            )