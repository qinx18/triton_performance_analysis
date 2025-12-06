import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program id and compute base offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential computation within each block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Compute im1 and im2 based on current index
        if idx == 0:
            im1 = n_elements - 1
            im2 = n_elements - 2
        elif idx == 1:
            im1 = 0
            im2 = n_elements - 1
        else:
            im1 = idx - 1
            im2 = idx - 2
            
        # Load values
        b_i = tl.load(b_ptr + idx)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute and store result
        result = (b_i + b_im1 + b_im2) * 0.333
        tl.store(a_ptr + idx, result)

def s292_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with one thread block per BLOCK_SIZE elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s292_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a