import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load data
    d_vals = tl.load(d_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Conditional computation
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    
    # Compute updates based on conditions
    update = tl.where(neg_mask, b_vals * c_vals,
                     tl.where(zero_mask, b_vals * b_vals, c_vals * c_vals))
    
    # Update a
    a_vals = a_vals + update
    
    # Store result
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )