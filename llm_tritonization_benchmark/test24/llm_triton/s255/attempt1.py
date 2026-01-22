import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel is inherently sequential due to carry-around variables
    # We need to process the entire array sequentially
    pid = tl.program_id(0)
    
    # Only the first program should execute
    if pid != 0:
        return
    
    # Initialize carry-around variables
    x = tl.load(b_ptr + (n - 1))
    y = tl.load(b_ptr + (n - 2))
    
    # Process the array sequentially in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load b values for this block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n:
                actual_offset = block_start + i
                b_val = tl.load(b_ptr + actual_offset)
                
                # Compute a[i] = (b[i] + x + y) * 0.333
                result = (b_val + x + y) * 0.333
                tl.store(a_ptr + actual_offset, result)
                
                # Update carry-around variables
                y = x
                x = b_val

def s255_triton(a, b, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program since computation is sequential
    grid = (1,)
    s255_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)