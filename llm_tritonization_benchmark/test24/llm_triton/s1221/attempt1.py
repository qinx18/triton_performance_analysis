import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(b_ptr, a_ptr, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Load from positions we need
    b_vals = tl.load(b_ptr + idx - 4)
    a_vals = tl.load(a_ptr + idx)
    
    # Compute result
    result = b_vals + a_vals
    
    # Store result
    tl.store(b_ptr + idx, result)

def s1221_triton(a, b):
    N = a.shape[0]
    
    # Strip size must be 4 due to dependency analysis
    STRIP_SIZE = 4
    BLOCK_SIZE = 4
    n_elements = N - 4
    
    # Process strips sequentially
    for strip_start in range(4, 4 + n_elements, STRIP_SIZE):
        remaining = min(STRIP_SIZE, 4 + n_elements - strip_start)
        if remaining <= 0:
            break
            
        # Launch kernel for this strip
        s1221_kernel[(1,)](b, a, strip_start, BLOCK_SIZE=remaining)