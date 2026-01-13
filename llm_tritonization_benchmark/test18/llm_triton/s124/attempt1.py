import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process elements sequentially in blocks
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    if block_start >= n_elements:
        return
        
    # Handle remaining elements in this block
    remaining = min(BLOCK_SIZE, n_elements - block_start)
    
    # Process each element in the block sequentially
    for i in range(remaining):
        idx = block_start + i
        
        # Load values for current element
        b_val = tl.load(b_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        e_val = tl.load(e_ptr + idx)
        
        # Compute result based on condition
        if b_val > 0.0:
            c_val = tl.load(b_ptr + idx)
        else:
            c_val = tl.load(c_ptr + idx)
            
        result = c_val + d_val * e_val
        
        # Store result at position idx (same as input position)
        tl.store(a_ptr + idx, result)

def s124_triton(a, b, c, d, e):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        N, BLOCK_SIZE
    )