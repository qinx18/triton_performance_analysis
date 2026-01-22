import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a, b, c, x_expanded, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    x_val = 0.0
    for i in range(n):
        a_val = tl.load(a + (n - 1 - i))
        b_val = tl.load(b + i)
        c_val = tl.load(c + i)
        x_val = a_val + b_val * c_val
        tl.store(x_expanded + i, x_val)

@triton.jit
def s281_phase1_kernel(a, a_copy, x_expanded, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < threshold
    
    x_vals = tl.load(x_expanded + offsets, mask=mask)
    tl.store(a + offsets, x_vals - 1.0, mask=mask)

@triton.jit
def s281_phase2_kernel(a, b, x_expanded, threshold, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    start_offsets = threshold + offsets
    mask = start_offsets < n
    
    x_vals = tl.load(x_expanded + start_offsets, mask=mask)
    tl.store(a + start_offsets, x_vals - 1.0, mask=mask)
    tl.store(b + start_offsets, x_vals, mask=mask)

@triton.jit
def s281_update_b_kernel(b, x_expanded, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < threshold
    
    x_vals = tl.load(x_expanded + offsets, mask=mask)
    tl.store(b + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    a_copy = a.clone()
    
    # Expand scalar x
    grid = (1,)
    s281_expand_x_kernel[grid](a, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 1: i = 0 to threshold-1
    grid = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid](a, a_copy, x_expanded, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Update b for first phase
    s281_update_b_kernel[grid](b, x_expanded, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    grid = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_phase2_kernel[grid](a, b, x_expanded, threshold, n, BLOCK_SIZE=BLOCK_SIZE)