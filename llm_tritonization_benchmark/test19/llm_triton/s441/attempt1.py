import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Conditional computation
    mask_negative = d_vals < 0.0
    mask_zero = d_vals == 0.0
    
    # Compute updates for each case
    update_negative = b_vals * c_vals
    update_zero = b_vals * b_vals
    update_positive = c_vals * c_vals
    
    # Select the appropriate update based on condition
    update = tl.where(mask_negative, update_negative,
                     tl.where(mask_zero, update_zero, update_positive))
    
    # Apply update
    result = a_vals + update
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)