import triton
import triton.language as tl
import torch

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements loop peeling with wrap around variable
    # The computation has dependencies that require sequential execution
    # We'll process one element at a time to maintain correctness
    
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one program handles the entire computation
        # Process elements sequentially to handle the im1 dependency
        for i in range(n_elements):
            # Calculate im1 based on position
            if i == 0:
                im1 = n_elements - 1
            else:
                im1 = i - 1
            
            # Load values
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            
            # Compute and store
            result = (b_i + b_im1) * 0.5
            tl.store(a_ptr + i, result)

def s291_triton(a, b):
    n_elements = a.numel()
    
    # Launch with single program since we need sequential processing
    BLOCK_SIZE = 32
    grid = (1,)
    
    s291_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a