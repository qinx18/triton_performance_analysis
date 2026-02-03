import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    
    # Create mask for valid indices (i < N)
    mask = offsets < N
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    N = a.shape[0]
    
    # We process indices 1 to N-1, so total elements to process is N-1
    elements_to_process = N - 1
    
    if elements_to_process <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s113_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )