import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Read from b[i-4] and a[i]
    b_vals = tl.load(b_ptr + idx - 4)
    a_vals = tl.load(a_ptr + idx)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_vals + a_vals
    
    # Store to b[i]
    tl.store(b_ptr + idx, result)

def s1221_triton(a, b):
    n = a.shape[0]
    
    # Process in strips of 4 to handle the RAW dependency
    STRIP_SIZE = 4
    n_elements = n - 4  # Elements to process (from i=4 to i=n-1)
    
    # Process each strip sequentially
    for strip_offset in range(0, n_elements, STRIP_SIZE):
        strip_start = 4 + strip_offset  # Actual starting index
        remaining = min(STRIP_SIZE, n - strip_start)
        
        if remaining > 0:
            grid = (1,)
            s1221_kernel[grid](a, b, strip_start, BLOCK_SIZE=remaining)