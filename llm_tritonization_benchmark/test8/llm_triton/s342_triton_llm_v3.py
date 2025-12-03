import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one block at a time sequentially
    # Cannot parallelize due to data dependency on j
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = -1
    
    # Process all elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            
            # Check if a[block_start + i] > 0
            if i < BLOCK_SIZE:
                a_val = tl.load(a_ptr + block_start + i)
                if a_val > 0.0:
                    j += 1
                    b_val = tl.load(b_ptr + j)
                    tl.store(a_ptr + block_start + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program instance since we need sequential execution
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a