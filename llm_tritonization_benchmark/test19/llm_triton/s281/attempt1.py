import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n // 2
    
    # Read from original arrays
    a_vals = tl.load(a_ptr + (n - 1 - offsets), mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x values
    x_vals = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)
    tl.store(x_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + n // 2
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Read from updated array
    a_vals = tl.load(a_ptr + (n - 1 - offsets), mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x values
    x_vals = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)
    tl.store(x_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Clone array for phase 1 reads
    a_orig = a.clone()
    
    # Phase 1: Process first half (i = 0 to n//2 - 1)
    grid1 = (triton.cdiv(n // 2, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a_orig, b, c, x, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Process second half (i = n//2 to n-1)
    grid2 = (triton.cdiv(n - n // 2, BLOCK_SIZE),)
    s281_phase2_kernel[grid2](a, b, c, x, n, BLOCK_SIZE=BLOCK_SIZE)