import triton
import triton.language as tl
import torch

@triton.jit
def s000_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < N
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = b[i] + 1
    a_vals = b_vals + 1.0
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s000_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    s000_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)