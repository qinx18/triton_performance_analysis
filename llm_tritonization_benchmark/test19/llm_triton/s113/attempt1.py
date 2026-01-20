import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust offsets to start from index 1 (skip a[0])
    offsets = offsets + 1
    
    # Create mask for valid indices
    mask = offsets < N
    
    # Load a[0] once (broadcast to all threads)
    a0 = tl.load(a_ptr)
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result back to a
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    N = a.shape[0]
    
    # We process indices 1 to N-1, so we need N-1 elements
    n_elements = N - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)