import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    for i in range(n):
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        x_val = a_val + b_val * c_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_compute_kernel(x_expanded_ptr, a_ptr, b_ptr, start_idx, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    indices = offsets + start_idx
    mask = indices < start_idx + BLOCK_SIZE
    
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    a_copy = a.clone()
    
    # Expand x values using original a values
    grid = (1,)
    s281_expand_x_kernel[grid](a_copy, b, c, x_expanded, n)
    
    # Phase 1: Process first half
    if threshold > 0:
        grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
        s281_compute_kernel[grid1](x_expanded, a, b, 0, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Process second half
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_compute_kernel[grid2](x_expanded, a, b, threshold, BLOCK_SIZE=BLOCK_SIZE)