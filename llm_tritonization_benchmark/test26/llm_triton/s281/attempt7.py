import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a, b, c, a_original, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load values
    a_vals = tl.load(a_original + (N - 1 - indices), mask=mask)
    b_vals = tl.load(b + indices, mask=mask)
    c_vals = tl.load(c + indices, mask=mask)
    
    # Compute x = a[N-1-i] + b[i] * c[i]
    x_vals = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(a + indices, x_vals - 1.0, mask=mask)
    tl.store(b + indices, x_vals, mask=mask)

def s281_triton(a, b, c):
    N = a.shape[0]
    threshold = N // 2
    BLOCK_SIZE = 256
    
    # Store original a values for reading
    a_original = a.clone()
    
    # Phase 1: Process first half (indices 0 to threshold-1)
    # These read from original values at high indices
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_kernel[grid1](a, b, c, a_original, threshold, BLOCK_SIZE)
    
    # Phase 2: Process second half (indices threshold to N-1) 
    # These read from updated values at low indices
    remaining = N - threshold
    grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
    
    # Offset pointers for second phase
    a_phase2 = a + threshold
    b_phase2 = b + threshold
    c_phase2 = c + threshold
    
    s281_kernel[grid2](a_phase2, b_phase2, c_phase2, a, remaining, BLOCK_SIZE)