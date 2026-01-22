import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel computes cumulative sum sequentially
    # Each thread handles one position to maintain dependency
    
    tid = tl.program_id(0)
    
    if tid < n:
        # Calculate cumulative sum up to position tid
        running_sum = 0.0
        
        # Sum all elements from 0 to tid (inclusive)
        for i in range(tid + 1):
            val = tl.load(a_ptr + i)
            running_sum += val
        
        # Store cumulative sum at position tid
        tl.store(b_ptr + tid, running_sum)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch one thread per element to handle sequential dependency
    grid = (n,)
    s3112_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return the final sum (last element of b contains the total sum)
    return b[-1].item()