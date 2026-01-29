import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load from read-only copy for reads
        a_vals = tl.load(a_copy_ptr + current_offsets + k, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Compute and store to original array
        result = a_vals + b_vals * c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n = a.shape[0] - 1
    if n <= 0:
        return
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s162_kernel[grid](
        a, a_copy, b, c, k, n, BLOCK_SIZE
    )