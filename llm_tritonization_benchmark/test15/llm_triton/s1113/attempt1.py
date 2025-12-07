import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel_phase1(a_ptr, b_ptr, orig_val, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = orig_val + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, threshold_idx, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    threshold_val = tl.load(a_ptr + threshold_idx)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = threshold_val + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Save original value at threshold before modification
    orig_a_at_threshold = a[threshold].clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold (inclusive) - uses original value
    grid1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel_phase1[grid1](
        a, b, orig_a_at_threshold, threshold + 1, BLOCK_SIZE
    )
    
    # Phase 2: i = threshold+1 to end - uses updated value
    if threshold + 1 < n:
        remaining = n - (threshold + 1)
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel_phase2[grid2](
            a[threshold + 1:], b[threshold + 1:], 0, remaining, BLOCK_SIZE
        )