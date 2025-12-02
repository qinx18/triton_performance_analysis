import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # This implements a simple pack operation that extracts positive values
    # Since this is inherently sequential, we'll use a single thread approach
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Use a single thread to process sequentially
    offsets = tl.arange(0, BLOCK_SIZE)
    j = 0
    
    # Process in blocks to handle large arrays
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b + current_offsets, mask=mask, other=0.0)
        
        # Check which values are positive
        pos_mask = b_vals > 0.0
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            if i < BLOCK_SIZE:
                # Extract the i-th element from the loaded block
                val = tl.load(b + block_start + i)
                if val > 0.0:
                    tl.store(a + j, val)
                    j += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single thread since this is inherently sequential
    grid = (1,)
    
    s341_kernel[grid](a, b, n_elements, BLOCK_SIZE)