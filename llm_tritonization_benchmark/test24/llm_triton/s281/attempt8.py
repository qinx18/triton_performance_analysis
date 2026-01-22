import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(a, b, c, x_val, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    reverse_offsets = n - 1 - offsets
    a_vals = tl.load(a + reverse_offsets, mask=mask)
    b_vals = tl.load(b + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x_computed = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a + offsets, x_computed - 1.0, mask=mask)
    tl.store(b + offsets, x_computed, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold-1 (reads original a values)
    a_copy = a.clone()
    grid = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel[grid](a_copy, b, c, x, threshold, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Copy phase 1 results back to original array
    a[:threshold] = a_copy[:threshold]
    
    # Phase 2: i = threshold to n-1 (reads updated a values) 
    remaining = n - threshold
    if remaining > 0:
        # Create shifted views for phase 2
        a_shifted = a[threshold:]
        b_shifted = b[threshold:]
        c_shifted = c[threshold:]
        
        # Need to adjust the kernel for phase 2 indexing
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_kernel[grid](a, b_shifted, c_shifted, x, remaining, remaining, BLOCK_SIZE=BLOCK_SIZE)