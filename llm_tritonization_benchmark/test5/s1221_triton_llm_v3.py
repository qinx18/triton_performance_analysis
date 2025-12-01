import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially to handle the dependency b[i] depends on b[i-4]
    for block_start in range(4, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Load b[i-4]
        b_prev_offsets = current_offsets - 4
        b_prev_vals = tl.load(a_ptr + b_prev_offsets)  # Always valid since block_start >= 4
        
        # Compute b[i] = b[i-4] + a[i]
        result = b_prev_vals + a_vals
        
        # Store result
        tl.store(b_ptr + current_offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program instance to handle dependencies
    s1221_kernel[(1,)](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return b