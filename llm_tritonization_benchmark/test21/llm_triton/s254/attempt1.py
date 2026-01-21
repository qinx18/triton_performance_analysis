import triton
import triton.language as tl
import torch

@triton.jit
def s254_expand_x_kernel(b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single block processes all elements sequentially to build expanded x array
    pid = tl.program_id(0)
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
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < N
    
    # Load b values and corresponding x values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    # Compute a[i] = (b[i] + x) * 0.5
    result = (b_vals + x_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded x values
    x_expanded = torch.zeros_like(b)
    
    # Phase 1: Expand scalar x to array using sequential processing
    grid = (1,)
    s254_expand_x_kernel[grid](b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Use expanded array in parallel computation
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s254_compute_kernel[grid](a, b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)