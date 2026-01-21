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
    
    # Load data
    a_vals = tl.load(a_ptr + (n - 1 - indices), mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute x for each element
    x_vals = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Clone array for reading original values
    a_original = a.clone()
    
    # Phase 1: Process first half (i = 0 to threshold-1)
    if threshold > 0:
        grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
        s281_kernel[grid1](a_original, b, c, threshold, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Process second half (i = threshold to n-1)
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        # Create offset pointers for the second phase
        a_offset = a + threshold
        b_offset = b + threshold
        c_offset = c + threshold
        s281_kernel[grid2](a_offset, b_offset, c_offset, remaining, remaining, BLOCK_SIZE=BLOCK_SIZE)