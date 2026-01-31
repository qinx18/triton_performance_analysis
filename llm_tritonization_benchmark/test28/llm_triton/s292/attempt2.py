import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    global_indices = block_start + offsets
    
    # Mask for valid elements
    mask = global_indices < n_elements
    
    # Compute im1 and im2 indices for each position
    im1 = tl.where(global_indices == 0, n_elements - 1, 
                   tl.where(global_indices == 1, 0, global_indices - 1))
    im2 = tl.where(global_indices == 0, n_elements - 2,
                   tl.where(global_indices == 1, n_elements - 1, global_indices - 2))
    
    # Load values
    b_i = tl.load(b_ptr + global_indices, mask=mask)
    b_im1 = tl.load(b_ptr + im1, mask=mask)
    b_im2 = tl.load(b_ptr + im2, mask=mask)
    
    # Compute result
    result = (b_i + b_im1 + b_im2) * 0.333
    
    # Store result
    tl.store(a_ptr + global_indices, result, mask=mask)

def s292_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )