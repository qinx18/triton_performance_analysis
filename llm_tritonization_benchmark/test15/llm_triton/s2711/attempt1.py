import torch
import triton
import triton.language as tl

@triton.jit
def s2711_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Check if b[i] != 0.0
    b_nonzero = b_vals != 0.0
    
    # Only load and compute where b is nonzero and within bounds
    compute_mask = mask & b_nonzero
    
    a_vals = tl.load(a_ptr + offsets, mask=compute_mask)
    c_vals = tl.load(c_ptr + offsets, mask=compute_mask)
    
    # Compute a[i] += b[i] * c[i] where b[i] != 0
    result = a_vals + b_vals * c_vals
    
    # Store back only where we computed
    tl.store(a_ptr + offsets, result, mask=compute_mask)

def s2711_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2711_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a