import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop for i = 0 to n-2
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load values for current iteration
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
    
    # Process S2 for all valid positions
    for block_start in range(1, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Get i-1 indices for loading b and d values
        prev_offsets = current_offsets - 1
        prev_mask = (current_offsets > 0) & mask
        
        # Load values needed for S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_next = tl.load(a_ptr + current_offsets, mask=mask)
        b_prev = tl.load(b_ptr + prev_offsets, mask=prev_mask)
        d_prev = tl.load(d_ptr + prev_offsets, mask=prev_mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_result = tl.where(prev_mask, b_prev + a_next * d_prev, a_next)
        tl.store(a_ptr + current_offsets, a_result, mask=mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s244_kernel[(1,)](
        a, b, c, d, n, BLOCK_SIZE
    )