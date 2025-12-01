import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    
    # Conditional computation
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    pos_mask = ~(neg_mask | zero_mask)
    
    # Compute updates for each condition
    update_neg = b_vals * c_vals
    update_zero = b_vals * b_vals
    update_pos = c_vals * c_vals
    
    # Apply conditional updates
    update = tl.where(neg_mask, update_neg, 
                     tl.where(zero_mask, update_zero, update_pos))
    
    # Update a
    result = a_vals + update
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )