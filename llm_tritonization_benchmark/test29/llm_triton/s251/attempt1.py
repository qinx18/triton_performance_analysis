import triton
import triton.language as tl
import torch

@triton.jit
def s251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for bounds checking
    mask = idx < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Compute: s = b[i] + c[i] * d[i], a[i] = s * s
    # Direct scalar expansion: replace s with the expression
    s_vals = b_vals + c_vals * d_vals
    a_vals = s_vals * s_vals
    
    # Store result
    tl.store(a_ptr + idx, a_vals, mask=mask)

def s251_triton(a, b, c, d):
    # Get array size from tensor shape
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s251_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )