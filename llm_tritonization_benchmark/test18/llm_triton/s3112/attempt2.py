import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single program processes all elements sequentially
    program_id = tl.program_id(0)
    
    # Only program 0 does the work
    if program_id == 0:
        running_sum = 0.0
        
        # Process in blocks
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            # Load current block
            vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            
            # Process each element sequentially within block
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    running_sum += vals[i]
                    # Store the cumulative sum
                    tl.store(b_ptr + block_start + i, running_sum)

def s3112_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = min(256, triton.next_power_of_2(n_elements))
    
    # Launch with single program
    grid = (1,)
    
    s3112_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the final sum (last element of b)
    return b[-1].item()