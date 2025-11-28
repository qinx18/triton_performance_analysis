import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Conditional computation
    cond1 = d_vals < 0.0
    cond2 = d_vals == 0.0
    
    # Compute updates based on conditions
    update1 = b_vals * c_vals  # d[i] < 0
    update2 = b_vals * b_vals  # d[i] == 0
    update3 = c_vals * c_vals  # d[i] > 0
    
    # Select appropriate update
    result = tl.where(cond1, update1, tl.where(cond2, update2, update3))
    
    # Add to existing values
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)