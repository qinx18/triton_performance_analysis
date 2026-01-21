import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Main loop - execute S0 and S1 for all iterations
    for block_start in range(0, N - 1, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N - 1
        
        # Load values
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        d_vals = tl.load(d_ptr + offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        new_b_vals = c_vals + b_vals
        tl.store(b_ptr + offsets, new_b_vals, mask=mask)
    
    # Epilogue - execute S2 only for the last iteration (i = N-2)
    if N >= 2:
        i = N - 2
        # Load values for the epilogue
        b_val = tl.load(b_ptr + i)
        a_next_val = tl.load(a_ptr + i + 1)
        d_val = tl.load(d_ptr + i)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_result = b_val + a_next_val * d_val
        tl.store(a_ptr + i + 1, a_result)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s244_kernel[grid](a, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE)