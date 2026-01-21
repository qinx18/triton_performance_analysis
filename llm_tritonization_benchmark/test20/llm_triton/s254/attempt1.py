import torch
import triton
import triton.language as tl

@triton.jit
def s254_expand_x_kernel(b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Expand scalar x to array using sequential processing"""
    pid = tl.program_id(0)
    
    # Only first thread processes everything sequentially
    if pid != 0:
        return
    
    # Initialize x with b[N-1]
    x_val = tl.load(b_ptr + N - 1)
    
    # Process all elements sequentially
    for i in range(N):
        # Store current x value
        tl.store(x_expanded_ptr + i, x_val)
        # Update x to b[i] for next iteration
        x_val = tl.load(b_ptr + i)

@triton.jit
def s254_compute_kernel(a_ptr, b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Compute a[i] = (b[i] + x_expanded[i]) * 0.5 in parallel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Compute a[i] = (b[i] + x) * 0.5
    result = (b_vals + x_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded x array
    x_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar x to array (sequential)
    grid = (1,)  # Single thread
    s254_expand_x_kernel[grid](b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute result in parallel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s254_compute_kernel[grid](a, b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)