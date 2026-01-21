import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for valid elements
    mask = idx < n
    
    # Calculate im1 and im2 for each position
    # For i=0: im1=n-1, im2=n-2
    # For i=1: im1=n-1, im2=0
    # For i>=2: im1=i-1, im2=i-2
    
    im1_indices = tl.where(idx == 0, n - 1,
                          tl.where(idx == 1, n - 1, idx - 1))
    
    im2_indices = tl.where(idx == 0, n - 2,
                          tl.where(idx == 1, 0, idx - 2))
    
    # Load values with masks
    b_i = tl.load(b_ptr + idx, mask=mask, other=0.0)
    b_im1 = tl.load(b_ptr + im1_indices, mask=mask, other=0.0)
    b_im2 = tl.load(b_ptr + im2_indices, mask=mask, other=0.0)
    
    # Compute result
    result = (b_i + b_im1 + b_im2) * 0.333
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s292_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s292_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)