import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Sequential processing within each block
    for block_offset in range(BLOCK_SIZE):
        i = block_start + block_offset
        if i >= n_elements:
            break
            
        # Compute im1 and im2 based on iteration i
        if i == 0:
            im1 = n_elements - 1
            im2 = n_elements - 2
        elif i == 1:
            im1 = 0
            im2 = n_elements - 1
        else:
            im1 = i - 1
            im2 = i - 2
        
        # Load values
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )