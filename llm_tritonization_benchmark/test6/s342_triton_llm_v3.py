import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a sequential unpacking operation
    # Each element processes the entire array sequentially
    pid = tl.program_id(0)
    
    # Only process with one thread block to maintain sequential order
    if pid != 0:
        return
    
    # Process sequentially in blocks
    j = -1
    
    # Pre-define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            # Check if a[block_start + i] > 0
            if block_start + i < n_elements:
                a_val = tl.load(a_ptr + block_start + i)
                if a_val > 0.0:
                    j += 1
                    # Load b[j] and store to a[block_start + i]
                    b_val = tl.load(b_ptr + j)
                    tl.store(a_ptr + block_start + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with single thread block to maintain sequential execution
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a