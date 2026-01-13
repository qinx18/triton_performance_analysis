import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S1 for all iterations except last
    for block_start in range(0, N - 2, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (N - 2)
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # S1: a[i] = b[i] + c[i]
        result = b_vals + c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)
    
    # Handle remaining elements for main loop
    remaining_start = ((N - 2) // BLOCK_SIZE) * BLOCK_SIZE
    if remaining_start < N - 2:
        current_offsets = remaining_start + offsets
        mask = current_offsets < (N - 2)
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        result = b_vals + c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)
    
    # Epilogue - execute S0 only for last iteration (i = N-2)
    if N >= 2:
        last_i = N - 2
        b_val = tl.load(b_ptr + last_i)
        e_val = tl.load(e_ptr + last_i)
        
        # S0: a[i+1] = b[i] + e[i] where i = N-2, so a[N-1] = b[N-2] + e[N-2]
        result = b_val + e_val
        tl.store(a_ptr + last_i + 1, result)

def s2244_triton(a, b, c, e):
    N = a.shape[0]
    
    if N < 2:
        return
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single program instance
    s2244_kernel[(1,)](
        a, b, c, e, N, BLOCK_SIZE
    )