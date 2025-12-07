import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(b_ptr, a_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 4  # +4 because loop starts from i=4
    
    mask = idx < n_elements
    
    # Load a[i] and b[i-4]
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx - 4, mask=mask)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_vals + a_vals
    
    # Store result
    tl.store(b_ptr + idx, result, mask=mask)

def s1221_triton(a, b):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 4  # Elements to process (i from 4 to LEN_1D-1)
    
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    
    # Process in strips sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_end = min(strip_start + STRIP_SIZE, n_elements)
        actual_strip_size = strip_end - strip_start
        
        if actual_strip_size > 0:
            # Launch kernel with single thread block per strip
            s1221_kernel[(1,)](
                b, a, 
                strip_start, 
                LEN_1D,
                BLOCK_SIZE=min(BLOCK_SIZE, actual_strip_size)
            )