import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        i_offsets = block_start + offsets
        
        # Check bounds and stride condition
        valid_mask = (i_offsets < n) & (i_offsets % inc == 0)
        read_valid_mask = valid_mask & ((i_offsets + inc) < (n + 1))
        
        # Load from read-only copy and original b
        a_copy_vals = tl.load(a_copy_ptr + i_offsets + inc, mask=read_valid_mask, other=0.0)
        b_vals = tl.load(b_ptr + i_offsets, mask=valid_mask, other=0.0)
        
        # Compute result
        result = a_copy_vals + b_vals
        
        # Store to original array
        tl.store(a_ptr + i_offsets, result, mask=valid_mask)

def s175_triton(a, b, inc):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n, BLOCK_SIZE=BLOCK_SIZE
    )