import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < (n // 2)
    
    # Read indices for a_copy
    read_indices = n - 1 - current_offsets
    
    # Load values
    a_vals = tl.load(a_copy_ptr + read_indices, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # Compute
    x = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + current_offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + current_offsets, x, mask=mask)

@triton.jit
def s281_kernel_phase2(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Phase 2 indices start from n//2
    phase2_indices = n // 2 + current_offsets
    mask = phase2_indices < n
    
    # Read indices for a (updated values from phase 1)
    read_indices = n - 1 - phase2_indices
    
    # Load values
    a_vals = tl.load(a_ptr + read_indices, mask=mask)
    b_vals = tl.load(b_ptr + phase2_indices, mask=mask)
    c_vals = tl.load(c_ptr + phase2_indices, mask=mask)
    
    # Compute
    x = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + phase2_indices, x - 1.0, mask=mask)
    tl.store(b_ptr + phase2_indices, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for Phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold-1
    grid_phase1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel_phase1[grid_phase1](
        a, a_copy, b, c, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    grid_phase2 = (triton.cdiv(remaining, BLOCK_SIZE),)
    s281_kernel_phase2[grid_phase2](
        a, b, c, n,
        BLOCK_SIZE=BLOCK_SIZE
    )