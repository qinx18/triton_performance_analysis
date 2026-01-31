import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Initialize x and y
        x = tl.load(b_ptr + n_elements - 1)
        y = tl.load(b_ptr + n_elements - 2)
        
        # Process all elements sequentially
        for i in range(n_elements):
            b_i = tl.load(b_ptr + i)
            result = (b_i + x + y) * 0.333
            tl.store(a_ptr + i, result)
            
            # Update x and y for next iteration
            y = x
            x = b_i

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    
    # Use single thread since computation is inherently sequential
    BLOCK_SIZE = 1
    grid = (1,)
    
    s255_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )