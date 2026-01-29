import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel_phase1(a_ptr, b_ptr, orig_value, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n) & (indices <= threshold)
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = orig_value + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n) & (indices > threshold)
    
    updated_value = tl.load(a_ptr + threshold)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = updated_value + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Save original value at threshold
    orig_value = a[threshold].clone()
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # Phase 1: i = 0 to threshold (inclusive)
    s1113_kernel_phase1[grid_size](
        a, b, orig_value, n, threshold, BLOCK_SIZE
    )
    
    # Phase 2: i = threshold+1 to end
    if threshold + 1 < n:
        s1113_kernel_phase2[grid_size](
            a, b, n, threshold, BLOCK_SIZE
        )