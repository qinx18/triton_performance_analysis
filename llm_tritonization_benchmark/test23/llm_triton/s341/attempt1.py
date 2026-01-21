import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a stream compaction operation that cannot be efficiently parallelized
    # We'll process sequentially in blocks, but the compaction itself is inherently serial
    
    pid = tl.program_id(0)
    if pid > 0:  # Only use first program
        return
    
    # Initialize counter
    j = -1
    
    # Process all elements sequentially
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for idx in range(BLOCK_SIZE):
            global_idx = block_start + idx
            if global_idx >= n:
                break
                
            # Load single element
            b_val = tl.load(b_ptr + global_idx)
            
            # Check condition and pack if positive
            if b_val > 0.0:
                j = j + 1
                tl.store(a_ptr + j, b_val)

def s341_triton(a, b):
    n = b.shape[0]
    BLOCK_SIZE = 256
    
    # Use only one thread block since this is inherently serial
    grid = (1,)
    
    s341_kernel[grid](
        b, a, n,
        BLOCK_SIZE=BLOCK_SIZE
    )