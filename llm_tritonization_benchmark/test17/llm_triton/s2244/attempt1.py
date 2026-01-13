import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Mask for valid elements (i < n_elements-1)
    mask = current_offsets < (n_elements - 1)
    
    # Load b[i], c[i], e[i]
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # S1: a[i] = b[i] + c[i] (execute for all iterations)
    a_vals = b_vals + c_vals
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # S0: a[i+1] = b[i] + e[i] (only execute for last iteration i = n_elements-2)
    last_iter_mask = mask & (current_offsets == (n_elements - 2))
    if tl.any(last_iter_mask):
        s0_vals = b_vals + e_vals
        tl.store(a_ptr + current_offsets + 1, s0_vals, mask=last_iter_mask)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Handle edge case
    if n_elements <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )