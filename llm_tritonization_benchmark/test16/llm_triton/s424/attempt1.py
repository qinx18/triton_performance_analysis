import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array, a, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid indices
    mask = idx < n_elements
    
    # Load from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array + idx, mask=mask)
    a_vals = tl.load(a + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1] = flat_2d_array[i+64]
    store_idx = idx + 64
    tl.store(flat_2d_array + store_idx, result, mask=mask)

def s424_triton(a, flat_2d_array):
    vl = 63
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in sequential strips of 64 elements
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_strip_size = strip_end - strip_start
        
        if current_strip_size > 0:
            # Use block size that matches strip size, but cap at 64
            current_block_size = min(BLOCK_SIZE, current_strip_size)
            
            # Round up to nearest power of 2 for Triton efficiency
            if current_block_size <= 32:
                current_block_size = 32
            else:
                current_block_size = 64
            
            grid = (1,)
            s424_kernel[grid](
                flat_2d_array, a, strip_start, strip_end,
                BLOCK_SIZE=current_block_size
            )