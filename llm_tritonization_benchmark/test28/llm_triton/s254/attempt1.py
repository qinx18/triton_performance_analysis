import torch
import triton
import triton.language as tl

@triton.jit
def s254_expand_x_kernel(b_ptr, x_expanded_ptr, N):
    # Single thread processes all elements sequentially to expand scalar x
    if tl.program_id(0) == 0:
        x_val = tl.load(b_ptr + (N - 1))  # x = b[LEN_1D-1]
        
        for i in range(N):
            tl.store(x_expanded_ptr + i, x_val)
            x_val = tl.load(b_ptr + i)  # x = b[i] for next iteration

@triton.jit
def s254_kernel(a_ptr, b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    result = (b_vals + x_vals) * 0.5
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    
    # Allocate expanded array for scalar x
    x_expanded = torch.zeros(N, dtype=torch.float32, device=a.device)
    
    # Phase 1: Expand scalar x to array
    grid = (1,)
    s254_expand_x_kernel[grid](b, x_expanded, N)
    
    # Phase 2: Compute result in parallel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s254_kernel[grid](a, b, x_expanded, N, BLOCK_SIZE)
    
    return a