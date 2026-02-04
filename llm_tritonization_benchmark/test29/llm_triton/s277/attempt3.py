import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n - 1)
    
    # Load values for this block
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Check first condition: a[i] >= 0
    a_positive = a_vals >= 0.0
    
    # Check second condition: b[i] >= 0 (only matters when a[i] < 0)
    b_positive = b_vals >= 0.0
    
    # Update a[i] when both a[i] < 0 and b[i] < 0
    should_update_a = (~a_positive) & (~b_positive)
    new_a = tl.where(should_update_a, a_vals + c_vals * d_vals, a_vals)
    tl.store(a_ptr + offsets, new_a, mask=mask)
    
    # Update b[i+1] when a[i] < 0 (regardless of b[i])
    should_update_b = ~a_positive
    b_next_vals = c_vals + d_vals * e_vals
    
    # Store b[i+1] values
    next_offsets = offsets + 1
    next_mask = mask & should_update_b & (next_offsets < n)
    tl.store(b_ptr + next_offsets, b_next_vals, mask=next_mask)

def s277_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    s277_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)