import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < threshold)
    
    # Read from a_copy[n-1-i], b[i], c[i]
    reverse_offsets = n - 1 - offsets
    reverse_mask = (reverse_offsets >= 0) & mask
    
    a_reverse = tl.load(a_copy_ptr + reverse_offsets, mask=reverse_mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x = a_reverse + b_vals * c_vals
    
    # Store a[i] = x - 1.0, b[i] = x
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n)
    
    # Read from a[n-1-i] (updated values from phase 1), b[i], c[i]
    reverse_offsets = n - 1 - offsets
    reverse_mask = (reverse_offsets >= 0) & mask
    
    a_reverse = tl.load(a_ptr + reverse_offsets, mask=reverse_mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x = a_reverse + b_vals * c_vals
    
    # Store a[i] = x - 1.0, b[i] = x
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = (n + 1) // 2
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold-1
    grid_phase1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid_phase1](
        a, a_copy, b, c, n, threshold, BLOCK_SIZE
    )
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    if remaining > 0:
        grid_phase2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid_phase2](
            a, b, c, n, threshold, BLOCK_SIZE
        )