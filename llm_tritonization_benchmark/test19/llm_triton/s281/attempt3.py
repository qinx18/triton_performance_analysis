import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_copy_ptr, a_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < threshold
    
    # Load data
    reverse_indices = n - 1 - indices
    reverse_mask = (reverse_indices >= 0) & (reverse_indices < n) & mask
    
    a_reverse = tl.load(a_copy_ptr + reverse_indices, mask=reverse_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Compute
    x = a_reverse + b_vals * c_vals
    
    # Store results
    a_new = x - 1.0
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, x, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load data
    reverse_indices = n - 1 - indices
    reverse_mask = (reverse_indices >= 0) & (reverse_indices < n) & mask
    
    a_reverse = tl.load(a_ptr + reverse_indices, mask=reverse_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Compute
    x = a_reverse + b_vals * c_vals
    
    # Store results
    a_new = x - 1.0
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, x, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Check if we have crossing threshold pattern
    threshold = n // 2
    
    if n > 1:  # Use two-phase approach for crossing threshold
        # Phase 1: Clone array for original values
        a_copy = a.clone()
        
        # Phase 1: indices 0 to threshold-1
        if threshold > 0:
            grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
            s281_phase1_kernel[grid1](a_copy, a, b, c, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
        
        # Phase 2: indices threshold to n-1
        remaining = n - threshold
        if remaining > 0:
            grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
            s281_phase2_kernel[grid2](a, b, c, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Single element case - handle manually
        if n == 1:
            x_val = a[0] + b[0] * c[0]
            a[0] = x_val - 1.0
            b[0] = x_val