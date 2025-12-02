import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    
    # Check if any d[i] < 0 - if so, we would exit in original code
    # In GPU context, we'll just skip computation for negative values
    valid_mask = mask & (d_vals >= 0.0)
    
    # Load other arrays only where valid
    a_vals = tl.load(a_ptr + block_start + offsets, mask=valid_mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=valid_mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=valid_mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result only where valid
    tl.store(a_ptr + block_start + offsets, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)