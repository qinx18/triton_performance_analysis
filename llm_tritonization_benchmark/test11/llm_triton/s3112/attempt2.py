import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential reduction that needs to be done by a single thread
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process elements sequentially
    running_sum = 0.0
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n_elements)
        block_size = block_end - block_start
        
        # Load current block
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = offsets < block_size
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in tl.static_range(BLOCK_SIZE):
            if i < block_size:
                running_sum += a_vals[i]
                # Store the running sum
                tl.store(b_ptr + (block_start + i), running_sum)

def s3112_triton(a, b):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s3112_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return b[-1].item()