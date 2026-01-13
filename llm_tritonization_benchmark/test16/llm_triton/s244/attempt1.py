import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process main loop iterations (i = 0 to n-2)
    current_offsets = block_start + offsets
    mask = current_offsets < (n - 1)
    
    # Load values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    a_next_vals = tl.load(a_ptr + current_offsets + 1, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (only for last iteration i = n-2)
    last_iter_mask = (current_offsets == (n - 2)) & mask
    a_next_new = b_new_vals + a_next_vals * d_vals
    tl.store(a_ptr + current_offsets + 1, a_next_new, mask=last_iter_mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    
    if n <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s244_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)