import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially to maintain the j increment logic
    # We use a single thread block to preserve the sequential nature of the induction variable
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize j to -1 (will be incremented before each use)
    j = -1
    
    # Process elements sequentially
    for i in range(n_elements):
        # Increment j for each element
        j += 1
        
        # Load individual elements
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Select value based on condition
        if b_val > 0.0:
            result = b_val + d_val * e_val
        else:
            result = c_val + d_val * e_val
        
        # Store result at position j
        tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single grid since we need sequential processing
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )