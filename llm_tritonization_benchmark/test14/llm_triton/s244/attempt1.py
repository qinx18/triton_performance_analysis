import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main computation for i in range(n_elements - 1)
    mask = (block_start + offsets) < (n_elements - 1)
    indices = block_start + offsets
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    a_next_vals = tl.load(a_copy_ptr + indices + 1, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + indices, b_new_vals, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (only for last iteration i = n_elements - 2)
    last_iter_mask = mask & (indices == (n_elements - 2))
    a_next_new = b_new_vals + a_next_vals * d_vals
    tl.store(a_ptr + indices + 1, a_next_new, mask=last_iter_mask)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, a_copy, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )