import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
    # Read from flat_2d_array[i]
    vals_flat = tl.load(flat_2d_array_ptr + idx, mask=mask)
    
    # Read from a[i]
    vals_a = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = vals_flat + vals_a
    
    # Write to xx[i+1] which is flat_2d_array[i+64]
    tl.store(flat_2d_array_ptr + idx + 64, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    vl = 63
    n_elements = a.shape[0] - 1
    
    STRIP_SIZE = 64
    
    # Process in strips sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        actual_strip_size = strip_end - strip_start
        
        if actual_strip_size <= 0:
            break
            
        # Find the largest power of 2 that doesn't exceed strip size
        block_size = 1
        while block_size * 2 <= actual_strip_size and block_size < 64:
            block_size *= 2
        
        # Launch kernel for this strip
        s424_kernel[(1,)](
            flat_2d_array,
            a,
            strip_end,
            strip_start,
            BLOCK_SIZE=block_size
        )