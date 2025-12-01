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
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=1.0)
    
    # Check if any d value is negative (this would cause exit in original)
    # In GPU context, we can't exit the entire program, so we'll skip computation
    # for threads where d[i] < 0
    valid_mask = mask & (d_vals >= 0.0)
    
    # Load b and c values
    b_vals = tl.load(b_ptr + offsets, mask=valid_mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=valid_mask, other=0.0)
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=valid_mask, other=0.0)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    
    # Check for negative values in d first (CPU check for early exit behavior)
    if torch.any(d < 0.0):
        # In the original C code, this would exit(0)
        # For GPU implementation, we'll raise an exception
        raise SystemExit("Negative value found in array d")
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a