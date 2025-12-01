import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[0] once (broadcast to all threads)
    a0 = tl.load(a_ptr)
    
    # Process elements starting from index 1
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i] values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = a[0] + b[i]
        result = a0 + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Launch kernel with single block (no grid needed for this pattern)
    s113_kernel[(1,)](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )