import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Phase 1: Save original a[0] value
    a_0_original = tl.load(a_ptr)
    
    # Phase 2: Update a[0] = a[0] + b[0]
    b_0 = tl.load(b_ptr)
    a_0_new = a_0_original + b_0
    tl.store(a_ptr, a_0_new)
    
    # Phase 3: Process remaining elements using updated a[0]
    for block_start in range(1, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        result = a_0_new + b_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s113_kernel[(1,)](a, b, n, BLOCK_SIZE)
    
    return a