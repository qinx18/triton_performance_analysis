import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Check if current_offsets + inc is within bounds
        read_mask = mask & ((current_offsets + inc) < (n + 1))
        
        # Load from read-only copy and b array
        a_vals = tl.load(a_copy_ptr + current_offsets + inc, mask=read_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute result
        result = a_vals + b_vals
        
        # Store to original array
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0] - 1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    s175_kernel[(1,)](
        a, a_copy, b, inc, n, BLOCK_SIZE
    )