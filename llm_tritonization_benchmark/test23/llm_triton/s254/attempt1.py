import triton
import triton.language as tl
import torch

@triton.jit
def s254_expand_x_kernel(x_expanded_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:  # Only first thread processes sequentially
        x_val = tl.load(b_ptr + n - 1)  # x = b[LEN_1D-1]
        
        for i in range(n):
            tl.store(x_expanded_ptr + i, x_val)
            x_val = tl.load(b_ptr + i)  # x = b[i] for next iteration

@triton.jit
def s254_compute_kernel(a_ptr, b_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    result = (b_vals + x_vals) * 0.5
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s254_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded array for x
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar x
    grid = (1,)  # Single thread
    s254_expand_x_kernel[grid](
        x_expanded, b, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s254_compute_kernel[grid](
        a, b, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE
    )