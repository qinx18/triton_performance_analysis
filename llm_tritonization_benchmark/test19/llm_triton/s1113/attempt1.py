import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load the constant value from a[n//2]
    threshold_idx = n // 2
    a_threshold = tl.load(a_ptr + threshold_idx)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[n//2] + b[i]
    result = a_threshold + b_vals
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Save original value at threshold before any modifications
    orig_a_at_threshold = a[threshold].clone()
    
    # Phase 1: Process elements 0 to threshold (uses original value)
    BLOCK_SIZE = 256
    grid_phase1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    
    # Create temporary arrays for phase 1
    a_phase1 = a[:threshold + 1]
    b_phase1 = b[:threshold + 1]
    
    s1113_phase1_kernel[grid_phase1](
        a_phase1, b_phase1, orig_a_at_threshold, threshold + 1, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Process elements threshold+1 to end (uses updated value)
    if threshold + 1 < n:
        remaining = n - (threshold + 1)
        grid_phase2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        
        # Get updated value at threshold
        updated_a_at_threshold = a[threshold]
        
        a_phase2 = a[threshold + 1:]
        b_phase2 = b[threshold + 1:]
        
        s1113_phase2_kernel[grid_phase2](
            a_phase2, b_phase2, updated_a_at_threshold, remaining, BLOCK_SIZE=BLOCK_SIZE
        )

@triton.jit
def s1113_phase1_kernel(a_ptr, b_ptr, orig_val, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = orig_a[threshold] + b[i]
    result = orig_val + b_vals
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)

@triton.jit
def s1113_phase2_kernel(a_ptr, b_ptr, updated_val, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = updated_a[threshold] + b[i]
    result = updated_val + b_vals
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)