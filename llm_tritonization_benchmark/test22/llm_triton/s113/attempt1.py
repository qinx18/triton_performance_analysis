import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    
    # Skip if n <= 1 (no elements to process)
    if n <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)  # n-1 elements to process (indices 1 to n-1)
    
    s113_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)