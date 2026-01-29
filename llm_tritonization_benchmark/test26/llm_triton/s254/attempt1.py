import triton
import triton.language as tl
import torch

@triton.jit
def s254_expand_kernel(b_ptr, x_expanded_ptr, n):
    # Single thread processes all elements sequentially
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    # Initialize x with b[n-1]
    x_val = tl.load(b_ptr + (n - 1))
    
    for i in range(n):
        # Store current x value
        tl.store(x_expanded_ptr + i, x_val)
        # Update x with b[i] for next iteration
        x_val = tl.load(b_ptr + i)

@triton.jit
def s254_kernel(a_ptr, b_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load b values and expanded x values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Compute a[i] = (b[i] + x) * 0.5
    a_vals = (b_vals + x_vals) * 0.5
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s254_triton(a, b):
    n = a.shape[0]
    
    # Create expanded x array
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Step 1: Expand scalar x
    s254_expand_kernel[(1,)](b, x_expanded, n)
    
    # Step 2: Compute in parallel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s254_kernel[grid](a, b, x_expanded, n, BLOCK_SIZE)
    
    return a