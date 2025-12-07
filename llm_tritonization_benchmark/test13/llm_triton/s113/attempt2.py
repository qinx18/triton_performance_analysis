import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load a[0] value that will be used for all computations
    a_0 = tl.load(a_ptr)
    
    # Process elements starting from index 1
    for block_start in range(1, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        result = a_0 + b_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s113_kernel[(1,)](a, b, n, BLOCK_SIZE)
    
    return a