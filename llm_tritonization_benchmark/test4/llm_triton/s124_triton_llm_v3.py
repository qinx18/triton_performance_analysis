import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(b_ptr, c_ptr, d_ptr, e_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input data
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute d[i] * e[i] once
        de_product = d_vals * e_vals
        
        # Apply conditional logic
        condition = b_vals > 0.0
        result = tl.where(condition, b_vals + de_product, c_vals + de_product)
        
        # Store results
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread block
    grid = (1,)
    s124_kernel[grid](
        b, c, d, e, a, n_elements, BLOCK_SIZE
    )
    
    return a