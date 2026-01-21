import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel needs to be sequential due to the dependency t = s
    # We'll process one element at a time using a single thread
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread block processes the entire computation
        t = 0.0
        
        # Process in blocks to handle large arrays efficiently
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, N, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N
            
            # Load b and c values
            b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
            c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
            
            # Process each element sequentially within the block
            for i in range(BLOCK_SIZE):
                if block_start + i < N:
                    s = b_vals[i] * c_vals[i]
                    a_val = s + t
                    tl.store(a_ptr + block_start + i, a_val)
                    t = s

def s252_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with single thread block since computation is sequential
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c, N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a