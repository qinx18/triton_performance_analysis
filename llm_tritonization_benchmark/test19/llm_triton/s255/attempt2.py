import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate the starting index for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load the initial carry-around variables
    x = tl.load(b_ptr + N - 1)
    y = tl.load(b_ptr + N - 2)
    
    # Process each element sequentially within this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < N:
            b_val = tl.load(b_ptr + idx)
            result = (b_val + x + y) * 0.333
            tl.store(a_ptr + idx, result)
            
            # Update carry-around variables
            y = x
            x = b_val

def s255_triton(a, b):
    N = a.shape[0]
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s255_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )