import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute statements S0 and S1 for all iterations
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        a_next_vals = tl.load(a_ptr + current_offsets + 1, mask=mask, other=0.0)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i] (use original b_vals)
        b_new_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i] (use updated b from S1)
        a_next_new = b_new_vals + a_next_vals * d_vals
        tl.store(a_ptr + current_offsets + 1, a_next_new, mask=mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s244_kernel[(1,)](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )