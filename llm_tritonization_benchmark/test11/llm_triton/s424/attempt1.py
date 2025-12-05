import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements
    mask = idx < n_elements
    
    # Load from flat_2d_array[idx] and a[idx]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[idx+1] which is flat_2d_array[idx+64]
    tl.store(flat_2d_array_ptr + idx + 64, result, mask=mask)

def s424_triton(a, flat_2d_array):
    STRIP_SIZE = 64
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    
    # Process in sequential strips
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_strip_size = strip_end - strip_start
        
        # Launch kernel for this strip
        num_blocks = triton.cdiv(current_strip_size, BLOCK_SIZE)
        
        for block_id in range(num_blocks):
            block_start = strip_start + block_id * BLOCK_SIZE
            elements_in_block = min(BLOCK_SIZE, strip_end - block_start)
            
            if elements_in_block > 0:
                s424_kernel[(1,)](
                    flat_2d_array, 
                    a, 
                    strip_end,
                    block_start,
                    BLOCK_SIZE=BLOCK_SIZE
                )
    
    return flat_2d_array