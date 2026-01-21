import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # if (d[i] < 0.)
    cond1 = d_vals < 0.0
    # else if (d[i] == 0.)
    cond2 = d_vals == 0.0
    # else (d[i] > 0.)
    cond3 = ~(cond1 | cond2)
    
    # Calculate updates for each condition
    update1 = b_vals * c_vals  # d < 0
    update2 = b_vals * b_vals  # d == 0
    update3 = c_vals * c_vals  # d > 0
    
    # Apply conditional updates
    result = a_vals + tl.where(cond1, update1, tl.where(cond2, update2, update3))
    
    tl.store(a_ptr + idx, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE)