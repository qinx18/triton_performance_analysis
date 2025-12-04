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
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # if (d[i] < 0.)
    cond1 = d_vals < 0.0
    # else if (d[i] == 0.)
    cond2 = d_vals == 0.0
    # else (d[i] > 0.)
    
    # Calculate all three possible operations
    op1 = b_vals * c_vals  # b[i] * c[i]
    op2 = b_vals * b_vals  # b[i] * b[i]
    op3 = c_vals * c_vals  # c[i] * c[i]
    
    # Select based on conditions
    result = tl.where(cond1, op1, tl.where(cond2, op2, op3))
    
    # Add to a[i]
    new_a = a_vals + result
    
    tl.store(a_ptr + indices, new_a, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )