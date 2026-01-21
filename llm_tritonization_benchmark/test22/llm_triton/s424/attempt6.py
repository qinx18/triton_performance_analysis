import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Read from flat_2d_array[i] and a[i]
    mask = offsets < n_elements
    flat_vals = tl.load(flat_2d_array_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to flat_2d_array[i + 64] (xx[i+1] where xx = flat_2d_array + 63)
    write_offsets = offsets + 64
    tl.store(flat_2d_array_ptr + write_offsets, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 64
    
    # Process in strips of 64 elements sequentially
    for strip_start in range(0, n_elements, BLOCK_SIZE):
        strip_end = min(strip_start + BLOCK_SIZE, n_elements)
        current_strip_size = strip_end - strip_start
        
        # Launch kernel for current strip
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array[strip_start:],
            a[strip_start:],
            current_strip_size,
            BLOCK_SIZE=BLOCK_SIZE
        )