import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # This kernel cannot be parallelized due to the dependency pattern
    # where each iteration depends on the previous value of im1
    # We need to handle this sequentially or use a different approach
    
    # For this specific pattern, we'll implement it as a single thread
    # processing all elements sequentially
    if pid == 0:
        for i in range(n_elements):
            if i == 0:
                im1 = n_elements - 1
            else:
                im1 = i - 1
            
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            result = (b_i + b_im1) * 0.5
            tl.store(a_ptr + i, result)

def s291_triton(a, b):
    n_elements = a.numel()
    
    # Launch with only one block since we need sequential processing
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s291_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a