import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Define offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S1 for all iterations (handles overwrite)
    for block_start in range(0, N - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (N - 1)
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # S1: a[i] = b[i] + c[i]
        result = b_vals + c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)
    
    # Epilogue - execute S0 only for last iteration
    # S0: a[i+1] = b[i] + e[i] at i = N-2
    if N >= 2:
        last_i = N - 2
        if tl.program_id(0) == 0:  # Only first program handles epilogue
            b_val = tl.load(b_ptr + last_i)
            e_val = tl.load(e_ptr + last_i)
            result = b_val + e_val
            tl.store(a_ptr + last_i + 1, result)

def s2244_triton(a, b, c, e):
    N = a.shape[0]
    
    if N < 2:
        return
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s2244_kernel[grid](
        a, b, c, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )