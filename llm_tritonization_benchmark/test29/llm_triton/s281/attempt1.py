import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, c_ptr, threshold, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < threshold) & (offsets < n)
    
    # Read indices for reverse access
    read_indices = n - 1 - offsets
    read_mask = mask & (read_indices >= 0) & (read_indices < n)
    
    # Load data
    a_vals = tl.load(a_copy_ptr + read_indices, mask=read_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

@triton.jit
def s281_kernel_phase2(a_ptr, b_ptr, c_ptr, threshold, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < n) & (offsets >= threshold)
    
    # Read indices for reverse access
    read_indices = n - 1 - offsets
    read_mask = mask & (read_indices >= 0) & (read_indices < n)
    
    # Load data (reading updated values from phase 1)
    a_vals = tl.load(a_ptr + read_indices, mask=read_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: process indices 0 to threshold-1
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel_phase1[grid1](
        a, a_copy, b, c, 
        threshold, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: process indices threshold to n-1
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_kernel_phase2[grid2](
            a, b, c,
            threshold, n,
            BLOCK_SIZE=BLOCK_SIZE
        )