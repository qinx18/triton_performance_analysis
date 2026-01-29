import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop iterations: i from 0 to N-2
    indices = block_start + offsets
    mask = indices < (N - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # S1: a[i] = b[i] + c[i] (executes for all iterations)
    result_s1 = b_vals + c_vals
    tl.store(a_ptr + indices, result_s1, mask=mask)
    
    # Epilogue - execute S0 only for the last iteration (i = N-2)
    # S0: a[i+1] = b[i] + e[i] where i = N-2
    last_i = N - 2
    if block_start <= last_i < block_start + BLOCK_SIZE:
        local_offset = last_i - block_start
        b_val = tl.load(b_ptr + last_i)
        e_val = tl.load(e_ptr + last_i)
        result_s0 = b_val + e_val
        tl.store(a_ptr + last_i + 1, result_s0)

def s2244_triton(a, b, c, e):
    N = a.shape[0]
    
    # Handle edge case where N < 2
    if N < 2:
        return
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(N - 1, BLOCK_SIZE)
    
    s2244_kernel[(num_blocks,)](
        a, b, c, e, N, BLOCK_SIZE
    )