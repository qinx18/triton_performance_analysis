import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets within the strip
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements in this strip
    mask = offsets < strip_size
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute xx[i+1] = flat_2d_array[i] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i+1] which is flat_2d_array[i+64]
    tl.store(xx_ptr + idx + 1, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Strip size is 64 due to dependency analysis
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Number of strips needed
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially to respect dependencies
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        remaining = strip_end - strip_start
        
        if remaining <= 0:
            break
            
        # Launch kernel for this strip only
        grid = (1,)
        
        # Manually create kernel for this strip
        s424_kernel[grid](
            flat_2d_array, a, xx, 
            strip_start, remaining,
            BLOCK_SIZE=BLOCK_SIZE
        )