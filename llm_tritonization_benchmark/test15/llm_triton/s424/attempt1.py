import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = idx < n_elements
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[i+1] = flat_2d_array[i+64+1] = flat_2d_array[i+65]
    write_idx = idx + 65
    write_mask = mask & (write_idx < (flat_2d_array_ptr.numel()))
    tl.store(flat_2d_array_ptr + write_idx, result, mask=write_mask)

def s424_triton(a, flat_2d_array):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process strips sequentially due to RAW dependency
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_block_size = strip_end - strip_start
        
        # Round up to nearest power of 2 for Triton
        triton_block_size = 1
        while triton_block_size < current_block_size:
            triton_block_size *= 2
        triton_block_size = min(triton_block_size, BLOCK_SIZE)
        
        s424_kernel[(1,)](
            flat_2d_array, a, n_elements, strip_start,
            BLOCK_SIZE=triton_block_size
        )