import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel_phase1(a_ptr, b_ptr, orig_val, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n) & (indices <= threshold)
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = orig_val + b_vals
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n) & (indices > threshold)
    
    a_threshold_val = tl.load(a_ptr + threshold)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = a_threshold_val + b_vals
    tl.store(a_ptr + indices, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Save original value at threshold before it gets modified
    orig_a_at_threshold = a[threshold].clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Phase 1: i = 0 to threshold (uses original value)
    s1113_kernel_phase1[grid](
        a, b, orig_a_at_threshold, n, threshold, BLOCK_SIZE
    )
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    s1113_kernel_phase2[grid](
        a, b, n, threshold, BLOCK_SIZE
    )