import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(
    a_ptr, b_ptr, c_ptr, e_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process main iterations (i = 0 to N-3)
    # Only execute S1 since S0 results will be overwritten
    for block_start in range(0, N - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (N - 1)
        
        # S1: a[i] = b[i] + c[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        result = b_vals + c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)
    
    # Epilogue: Execute S0 only for last iteration (i = N-2)
    if N >= 2:
        last_i = N - 2
        if tl.program_id(0) == 0:  # Execute on first program only
            # S0: a[i+1] = b[i] + e[i] where i = N-2
            b_val = tl.load(b_ptr + last_i)
            e_val = tl.load(e_ptr + last_i)
            result = b_val + e_val
            tl.store(a_ptr + (last_i + 1), result)

def s2244_triton(a, b, c, e):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )