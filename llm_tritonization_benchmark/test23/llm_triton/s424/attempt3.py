import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, strip_start, strip_len, BLOCK_SIZE: tl.constexpr):
    # Process one strip of up to 64 elements
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < strip_len
    
    # Load from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1]
    store_idx = idx + 1
    tl.store(xx_ptr + store_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1
    STRIP_SIZE = 64
    
    # Process strips sequentially to maintain dependencies
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_len = min(STRIP_SIZE, n_elements - strip_start)
        
        if strip_len <= 0:
            break
            
        # Use power-of-2 block size
        BLOCK_SIZE = 64
        
        # Launch single block for this strip
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array, 
            a, 
            xx, 
            strip_start,
            strip_len,
            BLOCK_SIZE=BLOCK_SIZE
        )