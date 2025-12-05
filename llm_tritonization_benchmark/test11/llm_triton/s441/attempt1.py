import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Conditional computation
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    pos_mask = (d_vals > 0.0) | ((d_vals != d_vals))  # positive or NaN
    
    # Compute updates for each case
    update1 = b_vals * c_vals  # d < 0
    update2 = b_vals * b_vals  # d == 0
    update3 = c_vals * c_vals  # d > 0
    
    # Select appropriate update
    result = tl.where(neg_mask, a_vals + update1,
                     tl.where(zero_mask, a_vals + update2, a_vals + update3))
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)