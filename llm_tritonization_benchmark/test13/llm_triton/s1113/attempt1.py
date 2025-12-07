import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel_phase1(a_ptr, b_ptr, orig_a_threshold, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < n) & (offsets <= threshold)
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    result = orig_a_threshold + b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < n) & (offsets > threshold)
    
    a_threshold = tl.load(a_ptr + threshold)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    result = a_threshold + b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Save original value before modification
    orig_a_threshold = a[threshold].clone()
    
    # Phase 1: i = 0 to threshold (uses original value)
    s1113_kernel_phase1[grid](
        a, b, orig_a_threshold, n, threshold, BLOCK_SIZE
    )
    
    # Phase 2: i = threshold+1 to end (uses updated value)  
    s1113_kernel_phase2[grid](
        a, b, n, threshold, BLOCK_SIZE
    )