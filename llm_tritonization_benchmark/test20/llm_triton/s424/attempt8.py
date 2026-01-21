import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    mask = offsets < strip_size
    
    # Read from flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Write to flat_2d_array[i+64] (which is xx[i+1])
    write_idx = idx + 63
    tl.store(flat_2d_array_ptr + write_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    n_elements = a.shape[0] - 1
    
    # Process in sequential strips to handle loop-carried dependency
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        # Launch kernel for this strip
        s424_kernel[(1,)](
            flat_2d_array, 
            a, 
            strip_start, 
            strip_size,
            BLOCK_SIZE=BLOCK_SIZE
        )