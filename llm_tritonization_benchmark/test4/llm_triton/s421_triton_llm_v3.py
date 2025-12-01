import torch
import triton
import triton.language as tl

@triton.jit
def s421_kernel(xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially to maintain dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load yy[i+1] values (shifted read)
        yy_offsets = current_offsets + 1
        yy_mask = yy_offsets < (n_elements + 1)  # Allow reading one element beyond for shift
        yy_vals = tl.load(xx_ptr + yy_offsets, mask=yy_mask, other=0.0)
        
        # Load a[i] values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute xx[i] = yy[i+1] + a[i]
        result = yy_vals + a_vals
        
        # Store result back to xx[i]
        tl.store(xx_ptr + current_offsets, result, mask=mask)

def s421_triton(xx, a):
    n_elements = len(a) - 1  # Process LEN_1D - 1 elements
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single thread block to maintain sequential dependencies
    
    s421_kernel[grid](
        xx, a, n_elements, BLOCK_SIZE
    )
    
    return xx