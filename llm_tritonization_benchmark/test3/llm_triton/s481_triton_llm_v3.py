import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d values and check for negative values
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Check if any d value is negative
    has_negative = tl.sum((d_vals < 0.0).to(tl.int32)) > 0
    
    # If any negative value found, we would need to exit
    # In GPU context, we skip computation for this block
    if has_negative:
        return
    
    # Load input arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Perform computation: a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    
    # Check for negative values on CPU first (since exit(0) would terminate)
    if torch.any(d < 0.0):
        import sys
        sys.exit(0)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a