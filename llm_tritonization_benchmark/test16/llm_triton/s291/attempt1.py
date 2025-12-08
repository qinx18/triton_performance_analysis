import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Compute offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Handle the sequential dependency using a loop
    # We need to process elements sequentially to maintain the im1 dependency
    for i in range(n_elements):
        if i == 0:
            im1_idx = n_elements - 1
        else:
            im1_idx = i - 1
            
        # Load b[i] and b[im1]
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1_idx)
        
        # Compute and store result
        result = (b_i + b_im1) * 0.5
        tl.store(a_ptr + i, result)

def s291_triton(a, b):
    n_elements = a.shape[0]
    
    # Since this has sequential dependencies, we use a single thread block
    # and process sequentially within the kernel
    BLOCK_SIZE = 1
    grid = (1,)
    
    s291_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a