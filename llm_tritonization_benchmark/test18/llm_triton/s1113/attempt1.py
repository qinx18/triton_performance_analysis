import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, n, threshold):
    pid = tl.program_id(0)
    BLOCK_SIZE = 256
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) <= threshold
    
    # Load original value at threshold
    a_threshold = tl.load(a_copy_ptr + threshold)
    
    # Load b values
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Compute and store
    result = a_threshold + b_vals
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, n, threshold):
    pid = tl.program_id(0)
    BLOCK_SIZE = 256
    
    block_start = (threshold + 1) + pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n
    
    # Load updated value at threshold
    a_threshold = tl.load(a_ptr + threshold)
    
    # Load b values
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    # Compute and store
    result = a_threshold + b_vals
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Create read-only copy before modification
    a_copy = a.clone()
    
    # Phase 1: i = 0 to threshold (inclusive)
    BLOCK_SIZE = 256
    grid1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel_phase1[grid1](a, a_copy, b, n, threshold)
    
    # Phase 2: i = threshold+1 to n-1
    if threshold + 1 < n:
        remaining = n - (threshold + 1)
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel_phase2[grid2](a, b, n, threshold)