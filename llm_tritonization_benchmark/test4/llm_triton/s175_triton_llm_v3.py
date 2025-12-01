import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, b_ptr, n, inc, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Check if indices are valid (i + inc < array bounds)
        read_mask = mask & ((current_offsets + inc) < (n + 1))
        
        # Load data
        a_read = tl.load(a_ptr + current_offsets + inc, mask=read_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute
        result = a_read + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    
    s175_kernel[(1,)](
        a, b, n, inc,
        BLOCK_SIZE=BLOCK_SIZE
    )