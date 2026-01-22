import torch
import triton
import triton.language as tl

@triton.jit
def s254_expand_x_kernel(b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    if tl.program_id(0) == 0:
        x_val = tl.load(b_ptr + N - 1)  # x = b[N-1]
        for i in range(N):
            tl.store(x_expanded_ptr + i, x_val)
            x_val = tl.load(b_ptr + i)  # x = b[i] for next iteration

@triton.jit
def s254_compute_kernel(a_ptr, b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    # Load b values and expanded x values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Compute a[i] = (b[i] + x) * 0.5
    result = (b_vals + x_vals) * 0.5
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array to store expanded x values
    x_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar x to array
    grid = (1,)
    s254_expand_x_kernel[grid](b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute results in parallel using expanded x
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s254_compute_kernel[grid](a, b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)