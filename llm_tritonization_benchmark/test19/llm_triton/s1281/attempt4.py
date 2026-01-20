import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(b_ptr, c_ptr, a_ptr, d_ptr, e_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    if tl.program_id(0) != 0:
        return
    
    x_val = 0.0
    for i in range(n):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        a_val = tl.load(a_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        x_val = b_val * c_val + a_val * d_val + e_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Compute new values
    a_new = x_vals - 1.0
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar x
    grid = (1,)
    expand_x_kernel[grid](b, c, a, d, e, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Use expanded array in parallel computation
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s1281_kernel[grid](a, b, c, d, e, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a, b