import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a sequential packing operation
    # It must be run with grid=(1,) to maintain sequential semantics
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = -1
    
    # Process elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            current_idx = block_start + i
            
            # Check bounds and skip if out of range
            if current_idx < n_elements:
                # Check if a[i] > 0
                a_val = tl.load(a_ptr + current_idx)
                if a_val > 0.0:
                    j += 1
                    # Load b[j] and store to a[i]
                    b_val = tl.load(b_ptr + j)
                    tl.store(a_ptr + current_idx, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block to maintain sequential execution
    grid = (1,)
    
    s342_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a