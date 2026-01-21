import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Compute x values for this block
    reverse_indices = n - 1 - indices
    reverse_mask = mask & (reverse_indices >= 0)
    
    a_vals = tl.load(a_ptr + reverse_indices, mask=reverse_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    x_vals = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Phase 1: Process first half with cloned array for reads
    if threshold > 0:
        a_copy = a.clone()
        grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
        # Create temporary views for phase 1
        a_phase1 = a[:threshold]
        a_copy_phase1 = a_copy[n-threshold:]
        b_phase1 = b[:threshold]
        c_phase1 = c[:threshold]
        
        # Manually compute phase 1 to ensure correct reverse indexing
        for i in range(threshold):
            x_val = a_copy[n-1-i] + b[i] * c[i]
            a[i] = x_val - 1.0
            b[i] = x_val
    
    # Phase 2: Process second half
    remaining = n - threshold
    if remaining > 0:
        # Manually compute phase 2 to use updated values from phase 1
        for i in range(threshold, n):
            x_val = a[n-1-i] + b[i] * c[i]
            a[i] = x_val - 1.0
            b[i] = x_val