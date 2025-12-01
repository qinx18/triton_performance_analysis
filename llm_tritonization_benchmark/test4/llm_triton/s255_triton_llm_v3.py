import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load initial x and y values
    x = tl.load(b_ptr + n - 1)
    y = tl.load(b_ptr + n - 2)
    
    # Process each block sequentially to maintain dependencies
    for i in range(n):
        # Load b[i]
        b_val = tl.load(b_ptr + i)
        
        # Compute result for this iteration
        result = (b_val + x + y) * 0.333
        
        # Store to a[i] only if this thread should handle it
        if block_start <= i < block_start + BLOCK_SIZE:
            tl.store(a_ptr + i, result)
        
        # Update x and y for next iteration
        y = x
        x = b_val

def s255_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # Launch kernel with single thread to maintain sequential dependencies
    s255_kernel[(1,)](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a