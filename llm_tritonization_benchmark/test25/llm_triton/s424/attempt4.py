import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = offsets < strip_size
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to xx[i+1] which is flat_2d_array[i+64+1]
    write_idx = idx + 64 + 1
    tl.store(flat_2d_array_ptr + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in strips sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        current_strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        if current_strip_size > 0:
            # Round up to nearest power of 2 for Triton
            triton_block_size = 1
            while triton_block_size < current_strip_size:
                triton_block_size *= 2
            triton_block_size = min(triton_block_size, BLOCK_SIZE)
            
            s424_kernel[(1,)](
                flat_2d_array,
                a,
                strip_start,
                current_strip_size,
                BLOCK_SIZE=triton_block_size
            )