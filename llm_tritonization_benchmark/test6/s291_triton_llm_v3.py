import triton
import triton.language as tl
import torch

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE
    
    # Process sequentially within each block to maintain dependencies
    for block_offset in range(BLOCK_SIZE):
        i = block_start + block_offset
        if i >= n_elements:
            break
            
        # Calculate im1 for current i
        if i == 0:
            im1 = n_elements - 1
        else:
            im1 = i - 1
            
        # Load b[i] and b[im1]
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1)
        
        # Compute and store result
        result = (b_i + b_im1) * 0.5
        tl.store(a_ptr + i, result)

def s291_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s291_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a