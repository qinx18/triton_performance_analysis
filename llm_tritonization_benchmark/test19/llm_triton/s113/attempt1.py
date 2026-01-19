import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    
    # Create offset vector
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices (i < n)
    mask = offsets < n
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size for indices 1 to n-1
    num_elements = n - 1
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)