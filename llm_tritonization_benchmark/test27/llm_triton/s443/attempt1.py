import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input arrays
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute condition: d[i] <= 0
    condition = d_vals <= 0.0
    
    # Compute both branches
    branch_bc = b_vals * c_vals  # b[i] * c[i] for d[i] <= 0
    branch_bb = b_vals * b_vals  # b[i] * b[i] for d[i] > 0
    
    # Select result based on condition
    result = tl.where(condition, branch_bc, branch_bb)
    
    # Update a[i] += result
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s443_triton(a, b, c, d):
    # Get array size from input tensor
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s443_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a