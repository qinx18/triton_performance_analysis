import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the sequential dependency by processing everything in one thread block
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize j and k
    j = 1
    k = 0
    
    # Sequential loop to handle dependencies
    i = n1 - 1
    while i < LEN_1D:
        k += j
        
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Load b[LEN_1D - k]
        b_idx = LEN_1D - k
        b_val = tl.load(b_ptr + b_idx)
        
        # Update a[i]
        tl.store(a_ptr + i, a_val + b_val)
        
        i += n3

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since we need sequential processing
    grid = (1,)
    
    s122_kernel[grid](
        a, b,
        n1, n3, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a