import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute statements S0 and S1 for all iterations
    for block_start in range(0, N - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (N - 1)
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
    
    # Epilogue - execute S2 only for the last iteration (i = N-2)
    if N >= 2:
        last_i = N - 2
        # Load the final values after S0 and S1 have been executed
        b_final = tl.load(b_ptr + last_i)
        a_next = tl.load(a_ptr + last_i + 1)
        d_final = tl.load(d_ptr + last_i)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_result = b_final + a_next * d_final
        tl.store(a_ptr + last_i + 1, a_result)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s244_kernel[grid](a, b, c, d, N, BLOCK_SIZE)