import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel_phase1(a, b, orig_value, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < threshold + 1)
    
    b_vals = tl.load(b + indices, mask=mask)
    result = orig_value + b_vals
    tl.store(a + indices, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a, b, updated_value, threshold, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + threshold + 1
    
    mask = (indices < n)
    
    b_vals = tl.load(b + indices, mask=mask)
    result = updated_value + b_vals
    tl.store(a + indices, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Save original value before it gets modified
    orig_a_at_threshold = a[threshold].clone()
    
    # Phase 1: i = 0 to threshold (uses original value)
    grid1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel_phase1[grid1](a, b, orig_a_at_threshold, threshold, BLOCK_SIZE)
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    if threshold + 1 < n:
        updated_a_at_threshold = a[threshold]
        remaining = n - (threshold + 1)
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel_phase2[grid2](a, b, updated_a_at_threshold, threshold, n, BLOCK_SIZE)