import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a sequential packing operation
    # We process one block at a time sequentially due to dependencies
    
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize j counter
    j = 0
    
    # Process all elements sequentially in chunks
    for start_idx in range(0, n, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        
        # Load current chunk of a
        a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Process each element in the chunk
        for local_i in range(BLOCK_SIZE):
            global_i = start_idx + local_i
            if global_i >= n:
                break
                
            # Load single element from a
            a_val = tl.load(a_ptr + global_i)
            
            # Check condition
            if a_val > 0.0:
                # Load from b[j] and store to a[i]
                b_val = tl.load(b_ptr + j)
                tl.store(a_ptr + global_i, b_val)
                j += 1

def s342_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single block to maintain sequential processing
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a