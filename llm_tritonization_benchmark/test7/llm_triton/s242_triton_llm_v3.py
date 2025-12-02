import torch
import triton
import triton.language as tl

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Process sequentially since each element depends on the previous one
    for i in range(1, n_elements):
        # Load previous value of a
        prev_val = tl.load(a_ptr + i - 1)
        
        # Load current values of b, c, d
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute new value
        new_val = prev_val + s1 + s2 + b_val + c_val + d_val
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s242_triton(a, b, c, d, s1, s2):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread since computation is sequential
    grid = (1,)
    s242_kernel[grid](
        a, b, c, d, 
        s1, s2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a