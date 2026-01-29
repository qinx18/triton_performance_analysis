import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < N
    
    # Load initial x and y values
    x = tl.load(b_ptr + (N - 1))
    y = tl.load(b_ptr + (N - 2))
    
    # Process elements in this block sequentially
    for i in range(block_start, min(block_start + BLOCK_SIZE, N)):
        # Load current b[i]
        b_i = tl.load(b_ptr + i)
        
        # Compute result
        result = (b_i + x + y) * 0.333
        
        # Store result
        tl.store(a_ptr + i, result)
        
        # Update carry variables
        y = x
        x = b_i

def s255_triton(a, b, x):
    N = a.shape[0]
    
    # Use small block size for sequential processing
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    s255_kernel[(num_blocks,)](
        a, b, N, BLOCK_SIZE
    )
    
    return a