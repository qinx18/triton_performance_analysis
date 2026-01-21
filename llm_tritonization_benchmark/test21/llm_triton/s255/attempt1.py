import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the block of elements this program will process
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load the block of b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load x and y values (last two elements of b)
    x = tl.load(b_ptr + N - 1)
    y = tl.load(b_ptr + N - 2)
    
    # Compute a[i] = (b[i] + x + y) * 0.333
    a_vals = (b_vals + x + y) * 0.333
    
    # Store the results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s255_triton(a, b, x):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s255_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return a