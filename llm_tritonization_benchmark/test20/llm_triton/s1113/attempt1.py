import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(a_ptr, b_ptr, n, threshold_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    result = threshold_val + b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Phase 1: i = 0 to threshold (uses original value)
    orig_a_at_threshold = a[threshold].clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel[grid](a, b, threshold + 1, orig_a_at_threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    remaining = n - threshold - 1
    if remaining > 0:
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel[grid](a[threshold + 1:], b[threshold + 1:], remaining, a[threshold], BLOCK_SIZE=BLOCK_SIZE)