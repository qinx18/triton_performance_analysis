import torch
import triton
import triton.language as tl

@triton.jit
def s254_expand_x_kernel(x_expanded_ptr, b_ptr, n_elements):
    """Phase 1: Expand scalar x to array using sequential processing"""
    # Single thread processes all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize x with b[n_elements-1]
    x_val = tl.load(b_ptr + n_elements - 1)
    
    for i in range(n_elements):
        # Store current x value
        tl.store(x_expanded_ptr + i, x_val)
        # Update x to b[i] for next iteration
        x_val = tl.load(b_ptr + i)

@triton.jit
def s254_kernel(a_ptr, b_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Phase 2: Compute a[i] = (b[i] + x_expanded[i]) * 0.5 in parallel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values and expanded x values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Compute a[i] = (b[i] + x_expanded[i]) * 0.5
    a_vals = (b_vals + x_vals) * 0.5
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s254_triton(a, b):
    n_elements = a.shape[0]
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar x to array
    grid = (1,)  # Single block for sequential processing
    s254_expand_x_kernel[grid](
        x_expanded, b, n_elements
    )
    
    # Phase 2: Compute a[i] in parallel using expanded x values
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s254_kernel[grid](
        a, b, x_expanded, n_elements, BLOCK_SIZE
    )