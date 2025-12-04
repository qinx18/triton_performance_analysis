import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential computation due to b[i] depending on b[i-4]
    # Process elements in order from index 4 to n_elements-1
    for start_idx in range(4, n_elements, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load current elements
        a_vals = tl.load(a_ptr + offsets, mask=mask)
        
        # Load b[i-4] values
        b_prev_offsets = offsets - 4
        b_prev_vals = tl.load(b_ptr + b_prev_offsets, mask=mask)
        
        # Compute b[i] = b[i-4] + a[i]
        b_new_vals = b_prev_vals + a_vals
        
        # Store results
        tl.store(b_ptr + offsets, b_new_vals, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread block for sequential execution
    s1221_kernel[(1,)](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return b