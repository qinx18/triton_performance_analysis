import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate starting index for this block (skip index 0)
    block_start = pid * BLOCK_SIZE + 1
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < n
    
    # Load a[0] (updated value)
    a0_val = tl.load(a_ptr)
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute results: a[i] = a[0] + b[i]
    results = a0_val + b_vals
    
    # Store results
    tl.store(a_ptr + indices, results, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    
    if n <= 1:
        return
    
    # Phase 1: Update a[0] = a[0] + b[0]
    a[0] = a[0] + b[0]
    
    # Phase 2: Handle i = 1 to n-1 in parallel using updated a[0]
    BLOCK_SIZE = 256
    remaining = n - 1
    if remaining > 0:
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s113_kernel[grid](a, b, n, BLOCK_SIZE)