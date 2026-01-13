import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Compute conditions
    cond_neg = d_vals < 0.0
    cond_zero = d_vals == 0.0
    
    # Compute updates based on conditions
    update_neg = b_vals * c_vals
    update_zero = b_vals * b_vals
    update_pos = c_vals * c_vals
    
    # Select appropriate update
    result = tl.where(cond_neg, a_vals + update_neg,
                     tl.where(cond_zero, a_vals + update_zero,
                             a_vals + update_pos))
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )