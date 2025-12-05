import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sequential packing operation
    # Since j depends on previous iterations, we need sequential processing
    
    # Initialize j counter
    j = -1
    
    # Process elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        # Calculate offsets for this block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load current block of a
        a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                # Check if a[i] > 0
                if a_vals[i] > 0.0:
                    j += 1
                    # Load b[j] and store to a[i]
                    b_val = tl.load(b_ptr + j)
                    # Create single-element mask and offset
                    single_offset = block_start + i
                    tl.store(a_ptr + single_offset, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    grid = (1,)  # Sequential processing requires single thread block
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a