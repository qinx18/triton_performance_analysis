import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_start = 0
    while block_start < (n - 1):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load a[i+1] and b[i]
        a_vals = tl.load(a_ptr + current_offsets + 1, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = a[i+1] + b[i]
        result = a_vals + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)
        
        block_start += BLOCK_SIZE

def s151_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s151_kernel[(1,)](
        a.data_ptr(), b.data_ptr(), n, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a