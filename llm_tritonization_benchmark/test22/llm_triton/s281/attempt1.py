import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    x_val = 0.0
    for i in range(n):
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        x_val = a_val + b_val * c_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < threshold) & (offsets >= 0)
    
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Update a and b arrays
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + threshold
    mask = (offsets < n) & (offsets >= threshold)
    
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Update a and b arrays
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    # Expand scalar x
    grid = (1,)
    s281_expand_x_kernel[grid](a, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 1: i = 0 to threshold-1
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, c, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](a, b, c, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)