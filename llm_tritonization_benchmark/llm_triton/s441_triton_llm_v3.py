import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute the three possible values
    val_neg = b_vals * c_vals
    val_zero = b_vals * b_vals
    val_pos = c_vals * c_vals
    
    # Create masks for the three conditions
    mask_neg = d_vals < 0.0
    mask_zero = d_vals == 0.0
    mask_pos = ~mask_neg & ~mask_zero
    
    # Select the appropriate value based on conditions
    result = tl.where(mask_neg, val_neg, tl.where(mask_zero, val_zero, val_pos))
    
    # Update a
    a_vals = a_vals + result
    
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )