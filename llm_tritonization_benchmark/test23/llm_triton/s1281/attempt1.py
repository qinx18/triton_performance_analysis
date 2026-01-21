import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    x_val = 0.0  # Initial value of x
    
    for i in range(N):
        # Compute x = b[i]*c[i] + a[i]*d[i] + e[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        a_val = tl.load(a_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        x_val = b_val * c_val + a_val * d_val + e_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Store results: a[i] = x - 1.0, b[i] = x
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded x array
    x_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar x sequentially
    grid = (1,)
    expand_x_kernel[grid](a, b, c, d, e, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Use expanded array in parallel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1281_kernel[grid](a, b, c, d, e, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)