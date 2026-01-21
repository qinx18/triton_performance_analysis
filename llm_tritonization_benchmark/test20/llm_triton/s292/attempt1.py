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
    
    # Process each element in the block
    for i in range(BLOCK_SIZE):
        current_idx = block_start + i
        if current_idx >= n:
            break
            
        # Calculate im1 and im2 for current iteration
        if current_idx == 0:
            im1 = n - 1
            im2 = n - 2
        elif current_idx == 1:
            im1 = n - 1
            im2 = 0
        else:
            im1 = current_idx - 1
            im2 = current_idx - 2
        
        # Load values
        b_i = tl.load(b_ptr + current_idx)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + current_idx, result)

def s292_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s292_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)