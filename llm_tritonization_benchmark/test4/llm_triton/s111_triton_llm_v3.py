import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n, BLOCK_SIZE * 2):
        current_offsets = block_start + offsets * 2
        mask = current_offsets < n
        
        # Load a[i-1] and b[i]
        prev_offsets = current_offsets - 1
        prev_mask = (current_offsets > 0) & mask
        
        a_prev = tl.load(a_ptr + prev_offsets, mask=prev_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = a[i-1] + b[i]
        result = a_prev + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s111_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 128
    
    s111_kernel[(1,)](a, b, n, BLOCK_SIZE)
    
    return a