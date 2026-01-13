import torch
import triton
import triton.language as tl

@triton.jit
def s254_expand_x_kernel(b_ptr, x_expanded_ptr, n_elements):
    # Single thread processes all elements sequentially
    if tl.program_id(0) != 0:
        return
    
    # Initialize x with b[n_elements-1]
    x_val = tl.load(b_ptr + n_elements - 1)
    
    for i in range(n_elements):
        # Store current x value
        tl.store(x_expanded_ptr + i, x_val)
        # Update x to b[i] for next iteration
        x_val = tl.load(b_ptr + i)

@triton.jit
def s254_compute_kernel(a_ptr, b_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    result = (b_vals + x_vals) * 0.5
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s254_triton(a, b):
    n_elements = a.shape[0]
    
    # Create expanded array for x
    x_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar x using sequential kernel
    grid = (1,)
    s254_expand_x_kernel[grid](b, x_expanded, n_elements)
    
    # Phase 2: Compute result in parallel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s254_compute_kernel[grid](a, b, x_expanded, n_elements, BLOCK_SIZE)