import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array, a, strip_start, strip_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Read from flat_2d_array[i] and a[i]
    mask = offsets < strip_size
    flat_vals = tl.load(flat_2d_array + idx, mask=mask)
    a_vals = tl.load(a + idx, mask=mask)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1] which is flat_2d_array[i+64]
    store_idx = idx + 64
    tl.store(flat_2d_array + store_idx, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    n_elements = N - 1
    
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process in strips of 64 elements sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        current_strip_size = strip_end - strip_start
        
        # Launch kernel for this strip
        s424_kernel[(1,)](
            flat_2d_array,
            a,
            strip_start,
            current_strip_size,
            BLOCK_SIZE=BLOCK_SIZE
        )