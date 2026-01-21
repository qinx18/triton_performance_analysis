import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n // 2
    
    # Load from copied array for reverse access
    rev_offsets = n - 1 - offsets
    a_vals = tl.load(a_copy_ptr + rev_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x values
    x_vals = a_vals + b_vals * c_vals
    
    # Update arrays
    a_new = x_vals - 1.0
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    threshold = n // 2
    block_start = pid * BLOCK_SIZE + threshold
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load from updated array for reverse access
    rev_offsets = n - 1 - offsets
    a_vals = tl.load(a_ptr + rev_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x values
    x_vals = a_vals + b_vals * c_vals
    
    # Update arrays
    a_new = x_vals - 1.0
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    # Phase 1: Handle first half where reads don't interfere with writes
    threshold = n // 2
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Handle second half where reads depend on phase 1 writes
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)