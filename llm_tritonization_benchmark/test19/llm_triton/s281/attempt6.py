import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = (indices < threshold) & (indices >= 0)
    
    reverse_indices = n - 1 - indices
    reverse_mask = mask & (reverse_indices >= 0) & (reverse_indices < n)
    
    a_vals = tl.load(a_copy_ptr + reverse_indices, mask=reverse_mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    x_vals = a_vals + b_vals * c_vals
    a_new = x_vals - 1.0
    
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = (indices < n) & (indices >= threshold)
    
    reverse_indices = n - 1 - indices
    reverse_mask = mask & (reverse_indices >= 0) & (reverse_indices < threshold)
    
    a_vals = tl.load(a_ptr + reverse_indices, mask=reverse_mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    x_vals = a_vals + b_vals * c_vals
    a_new = x_vals - 1.0
    
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, c, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2
    remaining = n - threshold
    grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_phase2_kernel[grid2](a, b, c, n, threshold, BLOCK_SIZE=BLOCK_SIZE)