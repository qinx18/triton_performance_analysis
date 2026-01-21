import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate global thread ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vectors
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S1 for all iterations
    current_offsets = block_start + offsets
    mask = current_offsets < (n - 1)
    
    # Load values for current block
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # S1: a[i] = b[i] + c[i]
    result_s1 = b_vals + c_vals
    tl.store(a_ptr + current_offsets, result_s1, mask=mask)
    
    # Epilogue - execute S0 only for last iteration (i = n-2)
    last_iter = n - 2
    last_mask = current_offsets == last_iter
    
    # Load values for S0 at last iteration
    b_last = tl.load(b_ptr + current_offsets, mask=last_mask)
    e_last = tl.load(e_ptr + current_offsets, mask=last_mask)
    
    # S0: a[i+1] = b[i] + e[i] (only at i = n-2, so a[n-1] = b[n-2] + e[n-2])
    result_s0 = b_last + e_last
    store_offsets = current_offsets + 1
    tl.store(a_ptr + store_offsets, result_s0, mask=last_mask)

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    
    if n <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e, n, BLOCK_SIZE=BLOCK_SIZE
    )