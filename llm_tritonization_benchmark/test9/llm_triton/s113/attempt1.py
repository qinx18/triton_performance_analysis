import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[0] once (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i] values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = a[0] + b[i]
        a_vals = a0 + b_vals
        
        # Store results
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s113_kernel[grid](a, b, n_elements, BLOCK_SIZE)