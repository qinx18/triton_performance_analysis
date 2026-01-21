import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes one strip of 64 elements
    strip_id = tl.program_id(0)
    strip_start = strip_id * 64
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask to ensure we don't go out of bounds
    mask = idx < n_elements
    
    # Load from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1] which is flat_2d_array[i+65] (since xx = flat_2d_array + 64)
    write_idx = idx + 65
    tl.store(xx_ptr + idx + 1, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = flat_2d_array.shape[0] - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Calculate number of strips needed
    n_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially to maintain dependencies
    for strip_id in range(n_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        strip_len = strip_end - strip_start
        
        if strip_len <= 0:
            break
            
        # Use block size that doesn't exceed strip length
        actual_block_size = min(BLOCK_SIZE, strip_len)
        
        # Launch single block for this strip
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array, 
            a, 
            xx, 
            strip_end,
            BLOCK_SIZE=actual_block_size
        )