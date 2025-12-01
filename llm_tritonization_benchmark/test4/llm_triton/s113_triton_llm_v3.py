import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[0] (broadcast to all threads)
    a0 = tl.load(a_ptr)
    
    for block_start in range(1, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load b[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = a[0] + b[i]
        result = a0 + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s113_kernel[grid](a, b, n, BLOCK_SIZE)
    
    return a