import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1  # Start from index 1
    
    # Create mask for valid indices (i >= 1 and i < N)
    mask = indices < N
    
    # Load a[0] (scalar broadcast)
    a_0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a_0 + b_vals
    
    # Store results back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s113_triton(a, b):
    N = a.shape[0]
    
    # We process indices 1 to N-1 (N-1 elements total)
    n_elements = N - 1
    
    if n_elements <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)