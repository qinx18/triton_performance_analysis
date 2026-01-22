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
def s281_phase1_kernel(a_copy_ptr, b_ptr, c_ptr, x_expanded_ptr, a_ptr, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < threshold
    
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + threshold
    mask = offsets < n
    
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    a_copy = a.clone()
    
    grid = (1,)
    s281_expand_x_kernel[grid](a_copy, b, c, x_expanded, n)
    
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a_copy, b, c, x_expanded, a, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](a, b, c, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)