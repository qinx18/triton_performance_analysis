import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel_phase1(a_ptr, orig_val, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = (current_offsets < n) & (current_offsets <= threshold)
    
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    result = orig_val + b_vals
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = (current_offsets < n) & (current_offsets > threshold)
    
    if tl.sum(mask.to(tl.int32)) > 0:
        updated_val = tl.load(a_ptr + threshold)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        result = updated_val + b_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    orig_a_at_threshold = a[threshold].clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1113_kernel_phase1[grid](
        a, orig_a_at_threshold, b, n, threshold, BLOCK_SIZE
    )
    
    s1113_kernel_phase2[grid](
        a, b, n, threshold, BLOCK_SIZE
    )