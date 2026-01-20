import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially, so use single thread
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first thread does all work
        # Initialize carry-around variables
        x = tl.load(b_ptr + N - 1)
        y = tl.load(b_ptr + N - 2)
        
        # Process all elements sequentially
        for i in range(N):
            b_val = tl.load(b_ptr + i)
            result = (b_val + x + y) * 0.333
            tl.store(a_ptr + i, result)
            
            # Update carry-around variables
            y = x
            x = b_val

def s255_triton(a, b, x):
    N = a.shape[0]
    
    # Use single block since computation must be sequential
    grid = (1,)
    BLOCK_SIZE = 1
    
    s255_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )