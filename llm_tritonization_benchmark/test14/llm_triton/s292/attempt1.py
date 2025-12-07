import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential computation - each thread block processes sequentially
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Process the entire array sequentially within one thread
    tid = tl.program_id(0) * BLOCK_SIZE + tl.program_id(1)
    if tid > 0:
        return
    
    # Initialize wrap-around variables
    im1 = n_elements - 1
    im2 = n_elements - 2
    
    # Process elements sequentially
    for i in range(n_elements):
        # Load current b[i]
        b_i = tl.load(b_ptr + i)
        
        # Load b[im1] and b[im2]
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + i, result)
        
        # Update wrap-around variables
        im2 = im1
        im1 = i

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single thread since computation must be sequential
    grid = (1, 1)
    
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a