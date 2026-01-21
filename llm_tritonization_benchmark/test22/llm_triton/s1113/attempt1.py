import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel_phase1(a_ptr, orig_val, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < threshold + 1) & (offsets < n)
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    result = orig_val + b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < n) & (offsets >= threshold + 1)
    
    updated_val = tl.load(a_ptr + threshold)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    result = updated_val + b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Save original value before it gets modified
    orig_a_at_threshold = a[threshold].clone()
    
    # Phase 1: i = 0 to threshold (uses original value)
    grid1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel_phase1[grid1](a, orig_a_at_threshold, b, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    remaining = n - threshold - 1
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel_phase2[grid2](a, b, n, threshold, BLOCK_SIZE=BLOCK_SIZE)