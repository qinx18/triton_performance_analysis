import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(
    a_ptr, b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Process each element sequentially to maintain dependencies
    x = tl.load(b_ptr + n_elements - 1)
    y = tl.load(b_ptr + n_elements - 2)
    
    for i in range(n_elements):
        # Load b[i]
        b_i = tl.load(b_ptr + i)
        
        # Compute a[i] = (b[i] + x + y) * 0.333
        result = (b_i + x + y) * 0.333
        
        # Store result
        tl.store(a_ptr + i, result)
        
        # Update x and y for next iteration
        y = x
        x = b_i

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    
    # Use single thread since we need sequential processing
    BLOCK_SIZE = 1
    grid = (1,)
    
    s255_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a