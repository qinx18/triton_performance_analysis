import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Conditional computation
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    pos_mask = (d_vals > 0.0)
    
    # Compute updates for each condition
    neg_update = b_vals * c_vals
    zero_update = b_vals * b_vals
    pos_update = c_vals * c_vals
    
    # Select appropriate update based on condition
    update = tl.where(neg_mask, neg_update, 
                     tl.where(zero_mask, zero_update, pos_update))
    
    # Apply update
    result = a_vals + update
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )