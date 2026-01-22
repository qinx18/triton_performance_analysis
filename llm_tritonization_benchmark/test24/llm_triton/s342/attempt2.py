import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the packing operation sequentially
    # since j depends on previous iterations
    
    # Process one block at a time sequentially
    block_id = tl.program_id(0)
    
    if block_id == 0:
        # Only first block does the work since it's inherently sequential
        j = -1
        
        # Process elements sequentially in chunks
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, n, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n
            
            # Load current block of a
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            
            # Process each element in the block
            for local_i in range(BLOCK_SIZE):
                global_i = block_start + local_i
                condition = global_i < n
                if condition:
                    # Extract scalar value for comparison
                    a_val = tl.load(a_ptr + global_i)
                    
                    if a_val > 0.0:
                        j = j + 1
                        b_val = tl.load(b_ptr + j)
                        tl.store(a_ptr + global_i, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since operation is sequential
    grid = (1,)
    s342_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a