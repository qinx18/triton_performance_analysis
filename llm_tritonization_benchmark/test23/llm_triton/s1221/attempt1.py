import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    strip_start = strip_id * 4
    
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
    n_elements = n - 4  # Start from i=4
    
    # Process in strips of 4 to handle the RAW dependency
    STRIP_SIZE = 4
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process each strip sequentially
    for strip_id in range(num_strips):
        strip_start = strip_id * STRIP_SIZE + 4  # +4 because we start from i=4
        remaining = min(STRIP_SIZE, n - strip_start)
        
        if remaining > 0:
            # Use block size equal to the remaining elements in this strip
            block_size = min(STRIP_SIZE, remaining)
            grid = (1,)
            
            # Adjust pointers to account for starting from i=4
            a_ptr = a[4:]
            b_ptr = b[4:]
            
            s1221_kernel[grid](a_ptr, b_ptr, BLOCK_SIZE=block_size)