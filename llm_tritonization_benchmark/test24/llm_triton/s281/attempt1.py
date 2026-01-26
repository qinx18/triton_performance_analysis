import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n // 2
    
    # Phase 1: i = 0 to n//2 - 1
    # Read from a_copy[n-1-i], write to a[i]
    read_offsets = n - 1 - offsets
    
    a_vals = tl.load(a_copy_ptr + read_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    x_vals = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = n // 2
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + base_offset
    mask = offsets < n
    
    # Phase 2: i = n//2 to n-1
    # Read from a[n-1-i] (updated values from Phase 1)
    read_offsets = n - 1 - offsets
    
    a_vals = tl.load(a_ptr + read_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    x_vals = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for Phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: Process first half
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, c, n, BLOCK_SIZE)
    
    # Phase 2: Process second half
    remaining = n - threshold
    grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_phase2_kernel[grid2](a, b, c, n, BLOCK_SIZE)