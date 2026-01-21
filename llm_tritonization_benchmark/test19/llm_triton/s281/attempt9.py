import triton
import triton.language as tl
import torch

@triton.jit
def s281_expand_x_kernel(a, b, c, x_expanded, n, BLOCK_SIZE: tl.constexpr):
    """Sequential kernel to expand scalar x with proper dependency handling"""
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
def s281_phase1_kernel(a, b, x_expanded, n, threshold, BLOCK_SIZE: tl.constexpr):
    """Phase 1: Process indices 0 to threshold-1"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < threshold
    
    x_vals = tl.load(x_expanded + offsets, mask=mask)
    tl.store(a + offsets, x_vals - 1.0, mask=mask)
    tl.store(b + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a, b, x_expanded, n, threshold, BLOCK_SIZE: tl.constexpr):
    """Phase 2: Process indices threshold to n-1"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + threshold
    mask = offsets < n
    
    x_vals = tl.load(x_expanded + offsets, mask=mask)
    tl.store(a + offsets, x_vals - 1.0, mask=mask)
    tl.store(b + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Step 1: Expand scalar x sequentially
    grid = (1,)
    s281_expand_x_kernel[grid](a, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 2: Phase 1 - parallel processing of first half
    grid = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid](a, b, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 3: Phase 2 - parallel processing of second half
    remaining = n - threshold
    if remaining > 0:
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid](a, b, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)