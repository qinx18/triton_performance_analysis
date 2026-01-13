import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < threshold)
    
    reverse_offsets = n - 1 - offsets
    
    a_vals = tl.load(a_copy_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    x = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n)
    
    reverse_offsets = n - 1 - offsets
    
    a_vals = tl.load(a_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    x = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: Process indices 0 to threshold-1
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, c, n, threshold, BLOCK_SIZE)
    
    # Phase 2: Process indices threshold to n-1
    remaining = n - threshold
    grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_phase2_kernel[grid2](a, b, c, n, threshold, BLOCK_SIZE)