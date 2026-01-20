import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid indices within strip
    mask = offsets < strip_size
    
    # Load from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1] which is flat_2d_array[i+64]
    write_idx = idx + 64
    tl.store(flat_2d_array_ptr + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially to maintain dependency
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        if strip_size <= 0:
            break
            
        # Launch single block for this strip
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array, 
            a, 
            strip_start,
            strip_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return flat_2d_array