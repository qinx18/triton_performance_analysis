import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes BLOCK_SIZE elements sequentially
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    if block_start >= n_elements:
        return
    
    # Calculate how many elements this block will process
    remaining = n_elements - block_start
    block_size = tl.minimum(remaining, BLOCK_SIZE)
    
    # Sequential processing within each block
    j = block_start - 1
    
    for i in range(block_size):
        idx = block_start + i
        j += 1
        
        # Load scalar values
        b_val = tl.load(b_ptr + idx)
        c_val = tl.load(c_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        e_val = tl.load(e_ptr + idx)
        
        # Compute result based on condition
        if b_val > 0.0:
            result = b_val + d_val * e_val
        else:
            result = c_val + d_val * e_val
        
        # Store result
        tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a