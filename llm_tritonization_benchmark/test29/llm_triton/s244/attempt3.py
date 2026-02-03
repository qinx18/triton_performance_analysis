import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop for all statements
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load original values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        a_next_vals = tl.load(a_ptr + current_offsets + 1, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_new = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i] (using updated b[i])
        a_next_new = b_new + a_next_vals * d_vals
        tl.store(a_ptr + current_offsets + 1, a_next_new, mask=mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s244_kernel[grid](a, b, c, d, n, BLOCK_SIZE)