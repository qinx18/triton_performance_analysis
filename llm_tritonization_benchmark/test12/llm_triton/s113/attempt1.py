import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Phase 1: Save original a[0] value
    if tl.program_id(0) == 0:
        orig_a0 = tl.load(a_ptr)
        # Store it temporarily at a[0] (will be overwritten)
        
        # Phase 2: Update a[1:] using original a[0]
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE + 1  # Start from index 1
        
        offsets = tl.arange(0, BLOCK_SIZE)
        indices = block_start + offsets
        mask = (indices < n) & (indices >= 1)
        
        b_vals = tl.load(b_ptr + indices, mask=mask)
        result = orig_a0 + b_vals
        
        tl.store(a_ptr + indices, result, mask=mask)

def s113_triton(a, b):
    n = a.shape[0]
    
    # Save original a[0] before modification
    orig_a0 = a[0].clone()
    
    # Calculate elements for indices 1 to n-1
    if n > 1:
        a[1:] = orig_a0 + b[1:]
    
    return a