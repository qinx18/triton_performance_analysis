import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    
    # Create offset vector
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices (must be >= 1 and < n)
    mask = offsets < n
    
    # Load a[0] (broadcast to all threads)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    
    # We start from index 1, so we need to process n-1 elements
    elements_to_process = n - 1
    
    if elements_to_process <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)