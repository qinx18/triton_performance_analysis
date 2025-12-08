import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n // 2
    
    # Phase 1: i = 0 to n//2 - 1
    # Read from a_copy[n-1-i], write to a[i]
    reverse_idx = n - 1 - idx
    
    a_vals = tl.load(a_copy_ptr + reverse_idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    x = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + idx, x - 1.0, mask=mask)
    tl.store(b_ptr + idx, x, mask=mask)

@triton.jit
def s281_kernel_phase2(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets + n // 2
    mask = idx < n
    
    # Phase 2: i = n//2 to n-1
    # Read from a[n-1-i] (updated values from phase 1), write to a[i]
    reverse_idx = n - 1 - idx
    
    a_vals = tl.load(a_ptr + reverse_idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    x = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + idx, x - 1.0, mask=mask)
    tl.store(b_ptr + idx, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for Phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: process first half
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel_phase1[grid1](a, a_copy, b, c, n, BLOCK_SIZE)
    
    # Phase 2: process second half
    grid2 = (triton.cdiv(n - threshold, BLOCK_SIZE),)
    s281_kernel_phase2[grid2](a, b, c, n, BLOCK_SIZE)