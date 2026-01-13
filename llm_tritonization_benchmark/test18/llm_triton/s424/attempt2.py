import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid indices within the strip
    mask = offsets < strip_size
    
    # Load from read position (idx + 0)
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to write position (idx + 64)
    tl.store(flat_2d_array_ptr + idx + 64, result, mask=mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Strip size is 64 (due to dependency analysis)
    STRIP_SIZE = 64
    
    # Process strips sequentially (due to loop-carried dependency)
    strip_start = 0
    while strip_start < n_elements:
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        
        if remaining > 0:
            BLOCK_SIZE = triton.next_power_of_2(remaining)
            # Launch kernel for this strip only
            s424_kernel[(1,)](
                flat_2d_array,
                a,
                strip_start,
                remaining,
                BLOCK_SIZE=BLOCK_SIZE
            )
        
        strip_start += STRIP_SIZE