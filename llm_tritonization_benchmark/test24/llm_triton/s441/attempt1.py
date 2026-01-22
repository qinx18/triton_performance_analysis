import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute conditions
    cond1 = d_vals < 0.0  # d[i] < 0
    cond2 = d_vals == 0.0  # d[i] == 0
    
    # Compute each branch
    branch1 = b_vals * c_vals  # b[i] * c[i]
    branch2 = b_vals * b_vals  # b[i] * b[i]
    branch3 = c_vals * c_vals  # c[i] * c[i]
    
    # Select based on conditions
    result = tl.where(cond1, branch1, tl.where(cond2, branch2, branch3))
    
    # Update a[i]
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s441_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d, N, 
        BLOCK_SIZE=BLOCK_SIZE
    )