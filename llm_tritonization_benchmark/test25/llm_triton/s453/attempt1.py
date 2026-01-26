import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Add 2.0 for each element in this block
        s += 2.0 * tl.sum(mask.to(tl.float32))
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = s * b[i] for each element
        a_vals = s * b_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s453_triton(a, b):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s453_kernel[grid](
        a, b,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )