import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = (block_start + offsets) < n_elements
    
    # Process each element in the block
    for idx in range(BLOCK_SIZE):
        global_idx = block_start + idx
        if global_idx >= n_elements:
            break
            
        # Calculate im1 and im2 based on current index
        if global_idx == 0:
            im1 = n_elements - 1
            im2 = n_elements - 2
        elif global_idx == 1:
            im1 = 0
            im2 = n_elements - 1
        else:
            im1 = global_idx - 1
            im2 = global_idx - 2
        
        # Load values
        b_i = tl.load(b_ptr + global_idx)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + global_idx, result)

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