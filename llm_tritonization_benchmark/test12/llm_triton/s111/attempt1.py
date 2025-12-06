import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, strip_start, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements
    mask = idx < n_elements
    
    # Read from a[i-1] and b[i]
    a_vals = tl.load(a_ptr + (idx - 1), mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i]
    tl.store(a_ptr + idx, result, mask=mask)

def s111_triton(a, b):
    # Start from index 1, step by 2, go up to LEN_1D
    n_elements = (a.shape[0] - 1 + 1) // 2  # Number of elements with step 2
    
    # Since we have loop-carried dependency, process strips of size 1 sequentially
    STRIP_SIZE = 1
    BLOCK_SIZE = 1
    
    # Process each strip sequentially
    for strip_id in range(n_elements):
        strip_start = 1 + strip_id * 2  # Start at 1, step by 2
        
        if strip_start < a.shape[0]:
            s111_kernel[(1,)](
                a, b, strip_start, a.shape[0], 
                BLOCK_SIZE=BLOCK_SIZE
            )