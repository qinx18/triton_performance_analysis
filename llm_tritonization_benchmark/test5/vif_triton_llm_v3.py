import torch
import triton
import triton.language as tl

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Create condition mask: b[i] > 0.0
    condition = b_vals > 0.0
    
    # Load current a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Apply conditional assignment: if b[i] > 0.0, then a[i] = b[i]
    result = tl.where(condition, b_vals, a_vals)
    
    # Store result back to a
    tl.store(a_ptr + offsets, result, mask=mask)

def vif_triton(a, b):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vif_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )