import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sequential dependency using a single block
    # Each thread handles one element sequentially
    
    # Only use block 0 for sequential processing
    if tl.program_id(0) != 0:
        return
    
    # Process elements sequentially to maintain dependency
    for i in range(n_elements):
        # Load elements
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute result based on condition
        de_product = d_val * e_val
        if b_val > 0.0:
            result = b_val + de_product
        else:
            result = c_val + de_product
        
        # Store result at position i (j starts at -1, so j++ makes it i)
        tl.store(a_ptr + i, result)

def s124_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Use a single block for sequential processing
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s124_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a