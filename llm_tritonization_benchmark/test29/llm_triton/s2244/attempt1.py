import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get thread block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S1 for all iterations
    current_offsets = block_start + offsets
    mask = current_offsets < (N - 1)
    
    # Load values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # S1: a[i] = b[i] + c[i]
    result_s1 = b_vals + c_vals
    tl.store(a_ptr + current_offsets, result_s1, mask=mask)
    
    # S0: a[i+1] = b[i] + e[i] (only for last iteration i = N-2)
    last_iter_mask = mask & (current_offsets == (N - 2))
    if tl.any(last_iter_mask):
        result_s0 = b_vals + e_vals
        tl.store(a_ptr + current_offsets + 1, result_s0, mask=last_iter_mask)

def s2244_triton(a, b, c, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(N - 1, BLOCK_SIZE)
    
    # Launch kernel
    s2244_kernel[(grid_size,)](
        a, b, c, e, N, BLOCK_SIZE
    )