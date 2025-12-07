import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n):
    BLOCK_SIZE = 512
    
    # Main loop - execute S1 for all iterations (no overwrite)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process main iterations
    current_offsets = block_start + offsets
    mask = current_offsets < (n - 1)
    
    # S1: a[i] = b[i] + c[i] (executed for all iterations)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    result_s1 = b_vals + c_vals
    tl.store(a_ptr + current_offsets, result_s1, mask=mask)
    
    # Epilogue - execute S0 only for last iteration (i = n-2)
    last_iter = n - 2
    last_mask = current_offsets == last_iter
    
    # Use sum to check if any element matches
    has_last = tl.sum(last_mask.to(tl.int32)) > 0
    
    if has_last:
        # S0: a[i+1] = b[i] + e[i] (only at i = n-2)
        b_val_last = tl.load(b_ptr + last_iter, mask=(last_iter >= 0))
        e_val_last = tl.load(e_ptr + last_iter, mask=(last_iter >= 0))
        result_s0 = b_val_last + e_val_last
        tl.store(a_ptr + last_iter + 1, result_s0, mask=(last_iter >= 0))

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    
    if n <= 1:
        return
    
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](a, b, c, e, n)