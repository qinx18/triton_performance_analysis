import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load a[0] once - it's broadcast to all elements
    a0 = tl.load(a_ptr)
    
    # Get block offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks starting from index 1
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i] values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = a[0] + b[i]
        result = a0 + b_vals
        
        # Store back to a[i]
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block since we're processing sequentially
    grid = (1,)
    s113_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)