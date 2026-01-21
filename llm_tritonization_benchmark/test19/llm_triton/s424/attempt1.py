import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid indices
    mask = idx < n_elements
    
    # Load from read positions
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to write positions (idx + 64)
    write_idx = idx + 64
    tl.store(flat_2d_array_ptr + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    n_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially to respect loop-carried dependencies
    for strip_id in range(n_strips):
        strip_start = strip_id * STRIP_SIZE
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        
        if strip_start >= n_elements:
            break
            
        current_strip_size = strip_end - strip_start
        
        # Launch kernel for this strip
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array,
            a,
            strip_start,
            strip_end,
            BLOCK_SIZE=BLOCK_SIZE
        )