import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Condition: if (i+1 < mid)
    cond_mask = (offsets + 1) < mid
    
    # Compute both branches
    branch1 = a_vals + b_vals * c_vals  # a[i] += b[i] * c[i]
    branch2 = a_vals + b_vals * d_vals  # a[i] += b[i] * d[i]
    
    # Select based on condition
    result = tl.where(cond_mask, branch1, branch2)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s276_triton(a, b, c, d, mid):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s276_kernel[grid](a, b, c, d, n, mid, BLOCK_SIZE=BLOCK_SIZE)