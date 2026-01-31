import triton
import triton.language as tl
import torch

@triton.jit
def s2711_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Check if b[i] != 0.0
    nonzero_mask = b_vals != 0.0
    compute_mask = mask & nonzero_mask
    
    # Load values only where we need to compute
    a_vals = tl.load(a_ptr + block_start + offsets, mask=compute_mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=compute_mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store back only where computation was needed
    tl.store(a_ptr + block_start + offsets, result, mask=compute_mask)

def s2711_triton(a, b, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2711_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)