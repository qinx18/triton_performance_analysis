import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Phase 1: i = 0 (uses original value)
    orig_val = tl.load(a_copy_ptr)
    mask = offsets < 1
    tl.store(a_ptr + offsets, orig_val, mask=mask)
    
    # Phase 2: i = 1 to end (uses updated value from Phase 1)
    for block_start in range(1, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load the updated value from a[0]
        updated_val = tl.load(a_ptr)
        
        # Store to all positions in this block
        tl.store(a_ptr + current_offsets, updated_val, mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Create read-only copy before kernel launch
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s293_kernel[grid](a, a_copy, n, BLOCK_SIZE)