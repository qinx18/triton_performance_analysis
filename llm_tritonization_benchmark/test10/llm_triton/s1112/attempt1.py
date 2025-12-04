import torch
import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks from high to low addresses to preserve loop reversal semantics
    for block_start in range(((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE - 1) * BLOCK_SIZE, -1, -BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = b[i] + 1.0
        a_vals = b_vals + 1.0
        
        # Store result
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s1112_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s1112_kernel[(1,)](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a