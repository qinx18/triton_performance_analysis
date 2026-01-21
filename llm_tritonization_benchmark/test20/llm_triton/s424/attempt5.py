import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = offsets < strip_size
    
    # Read from flat_2d_array[i] and a[i]
    vals_flat = tl.load(flat_2d_array_ptr + idx, mask=mask)
    vals_a = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = vals_flat + vals_a
    
    # Write to flat_2d_array[i+64] (which is xx[i+1] since xx = flat_2d_array + 63)
    write_idx = idx + 64
    tl.store(flat_2d_array_ptr + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1
    STRIP_SIZE = 64
    
    # Process in strips sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        if strip_size <= 0:
            break
            
        # Use power-of-2 block size that doesn't exceed strip size
        block_size = 1
        while block_size * 2 <= strip_size and block_size < 1024:
            block_size *= 2
        
        # Launch kernel for this strip
        s424_kernel[(1,)](
            flat_2d_array,
            a,
            strip_start,
            strip_size,
            BLOCK_SIZE=block_size
        )