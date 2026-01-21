import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Main loop - execute S0 and S1 for all iterations
    for block_start in range(0, n - 1, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n - 1)
        
        # Load values
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        d_vals = tl.load(d_ptr + offsets, mask=mask)
        a_next_vals = tl.load(a_ptr + offsets + 1, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + offsets, b_new, mask=mask)
    
    # Epilogue - execute S2 only for last iteration (i = n-2)
    if n >= 2:
        i = n - 2
        b_last = tl.load(b_ptr + i)
        a_last = tl.load(a_ptr + i + 1)
        d_last = tl.load(d_ptr + i)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_result = b_last + a_last * d_last
        tl.store(a_ptr + i + 1, a_result)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s244_kernel[(1,)](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)