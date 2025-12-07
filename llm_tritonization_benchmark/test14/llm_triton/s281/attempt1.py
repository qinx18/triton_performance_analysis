import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n // 2
    
    # For phase 1: i goes from 0 to n//2 - 1
    # Read from a_copy[n-1-i], write to a[i]
    reverse_indices = (n - 1) - indices
    
    # Load values
    a_vals = tl.load(a_copy_ptr + reverse_indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute
    x = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + indices, x - 1.0, mask=mask)
    tl.store(b_ptr + indices, x, mask=mask)

@triton.jit  
def s281_kernel_phase2(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + n // 2
    mask = indices < n
    
    # For phase 2: i goes from n//2 to n-1
    # Read from a[n-1-i] (updated values from phase 1), write to a[i]
    reverse_indices = (n - 1) - indices
    
    # Load values
    a_vals = tl.load(a_ptr + reverse_indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute
    x = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + indices, x - 1.0, mask=mask)
    tl.store(b_ptr + indices, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    # Phase 1: process indices 0 to threshold-1
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel_phase1[grid1](a, a_copy, b, c, n, BLOCK_SIZE)
    
    # Phase 2: process indices threshold to n-1
    remaining = n - threshold
    grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_kernel_phase2[grid2](a, b, c, n, BLOCK_SIZE)