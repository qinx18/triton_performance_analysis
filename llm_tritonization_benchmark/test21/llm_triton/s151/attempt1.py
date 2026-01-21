import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate which block this program handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (n-1 because loop goes to LEN_1D-1)
    mask = offsets < (n - 1)
    
    # Load a[i+1] (shift by 1) and b[i]
    a_shifted = tl.load(a_ptr + offsets + 1, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_shifted + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s151_triton(a, b):
    n = a.shape[0]
    
    # Use block size of 256
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    # Launch kernel
    s151_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)