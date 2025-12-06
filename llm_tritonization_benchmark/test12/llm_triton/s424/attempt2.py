import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[i+1] which is flat_2d_array[i+1+63] = flat_2d_array[i+64]
    write_idx = idx + 64
    write_mask = mask & (write_idx < 65536)  # flat_2d_array max size
    tl.store(flat_2d_array_ptr + write_idx, result, mask=write_mask)

def s424_triton(a, flat_2d_array):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in strips sequentially
    strip_start = 0
    while strip_start < n_elements:
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_strip_size = strip_end - strip_start
        
        # Use appropriate block size for this strip
        current_block_size = min(BLOCK_SIZE, current_strip_size)
        
        # Round up to nearest power of 2 for Triton efficiency
        triton_block_size = 1
        while triton_block_size < current_block_size:
            triton_block_size *= 2
        triton_block_size = min(triton_block_size, 64)
        
        s424_kernel[(1,)](
            flat_2d_array, 
            a, 
            strip_start, 
            strip_end,
            BLOCK_SIZE=triton_block_size
        )
        
        strip_start += STRIP_SIZE