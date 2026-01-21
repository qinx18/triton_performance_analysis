import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate grid position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S1 for all iterations
    current_offsets = block_start + offsets
    mask = current_offsets < (N - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # S1: a[i] = b[i] + c[i] (for all iterations)
    result_s1 = b_vals + c_vals
    tl.store(a_ptr + current_offsets, result_s1, mask=mask)
    
    # Epilogue - execute S0 only for last iteration (i = N-2)
    # S0: a[i+1] = b[i] + e[i] only at i = N-2
    last_iter_mask = current_offsets == (N - 2)
    final_mask = mask & last_iter_mask
    
    # For i = N-2, compute a[N-1] = b[N-2] + e[N-2]
    result_s0 = b_vals + e_vals
    tl.store(a_ptr + current_offsets + 1, result_s0, mask=final_mask)

def s2244_triton(a, b, c, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    s2244_kernel[grid](a, b, c, e, N, BLOCK_SIZE=BLOCK_SIZE)