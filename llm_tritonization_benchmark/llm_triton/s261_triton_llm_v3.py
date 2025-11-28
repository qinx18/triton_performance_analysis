import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Handle i starting from 1, so adjust offsets
    i_offsets = offsets + 1
    i_mask = i_offsets < n_elements
    
    # Load data for current positions
    a_vals = tl.load(a_ptr + i_offsets, mask=i_mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=i_mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=i_mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=i_mask)
    
    # Load c[i-1] values
    c_prev_offsets = i_offsets - 1
    c_prev_mask = (c_prev_offsets >= 0) & i_mask
    c_prev_vals = tl.load(c_ptr + c_prev_offsets, mask=c_prev_mask)
    
    # Compute: t = a[i] + b[i]
    t1 = a_vals + b_vals
    
    # Compute: a[i] = t + c[i-1]
    a_new = t1 + c_prev_vals
    
    # Store updated a values
    tl.store(a_ptr + i_offsets, a_new, mask=i_mask)
    
    # Compute: t = c[i] * d[i]
    t2 = c_vals * d_vals
    
    # Store: c[i] = t
    tl.store(c_ptr + i_offsets, t2, mask=i_mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Since we start from i=1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )