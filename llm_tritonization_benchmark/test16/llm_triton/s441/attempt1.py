import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Create condition masks
    d_negative = d_vals < 0.0
    d_zero = d_vals == 0.0
    
    # Compute updates for each condition
    update_negative = b_vals * c_vals
    update_zero = b_vals * b_vals
    update_positive = c_vals * c_vals
    
    # Select appropriate update based on conditions
    update = tl.where(d_negative, update_negative,
                     tl.where(d_zero, update_zero, update_positive))
    
    # Apply update
    result = a_vals + update
    
    tl.store(a_ptr + idx, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)