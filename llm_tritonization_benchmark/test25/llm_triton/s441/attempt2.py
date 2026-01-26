import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Conditional logic
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    
    # Compute updates based on conditions
    update1 = b_vals * c_vals  # d < 0
    update2 = b_vals * b_vals  # d == 0  
    update3 = c_vals * c_vals  # d > 0
    
    # Apply conditional updates
    result = tl.where(neg_mask, a_vals + update1, 
                     tl.where(zero_mask, a_vals + update2, a_vals + update3))
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s441_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )