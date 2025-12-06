import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Conditional computation
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    pos_mask = (d_vals > 0.0) | ((d_vals != d_vals) & ~neg_mask & ~zero_mask)  # positive or NaN case
    
    # Compute updates for each condition
    update1 = b_vals * c_vals  # d[i] < 0
    update2 = b_vals * b_vals  # d[i] == 0
    update3 = c_vals * c_vals  # d[i] > 0
    
    # Select the appropriate update based on conditions
    result = tl.where(neg_mask, a_vals + update1,
                     tl.where(zero_mask, a_vals + update2, a_vals + update3))
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )