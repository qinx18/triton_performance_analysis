import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets starting from 1 (skip index 0)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (must be < N and >= 1)
    mask = offsets < N
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    N = a.shape[0]
    
    # Only process indices 1 to N-1 (skip index 0)
    if N <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)