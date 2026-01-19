import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel(a_ptr, b_ptr, n, threshold_value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    result = b_vals + threshold_value
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Phase 1: i = 0 to threshold (uses original value)
    orig_a_at_threshold = a[threshold].item()
    
    BLOCK_SIZE = 256
    grid_phase1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel[grid_phase1](a, b, threshold + 1, orig_a_at_threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    if threshold + 1 < n:
        remaining = n - (threshold + 1)
        a_slice = a[threshold + 1:]
        b_slice = b[threshold + 1:]
        updated_a_at_threshold = a[threshold].item()
        
        grid_phase2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel[grid_phase2](a_slice, b_slice, remaining, updated_a_at_threshold, BLOCK_SIZE=BLOCK_SIZE)