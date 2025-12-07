import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(a_ptr, b_ptr, n, threshold, a_threshold_val, phase, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        if phase == 1:
            # Phase 1: i = 0 to threshold (inclusive)
            phase_mask = mask & (current_offsets <= threshold)
        else:
            # Phase 2: i = threshold+1 to end
            phase_mask = mask & (current_offsets > threshold)
        
        if tl.sum(phase_mask.to(tl.int32)) > 0:
            # Load b values
            b_vals = tl.load(b_ptr + current_offsets, mask=phase_mask)
            
            # Compute result
            result = a_threshold_val + b_vals
            
            # Store to a
            tl.store(a_ptr + current_offsets, result, mask=phase_mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Save original value at threshold
    orig_a_threshold = a[threshold].clone()
    
    # Phase 1: i = 0 to threshold (uses original value)
    s1113_kernel[grid](
        a,
        b,
        n,
        threshold,
        orig_a_threshold,
        1,
        BLOCK_SIZE
    )
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    updated_a_threshold = a[threshold]
    s1113_kernel[grid](
        a,
        b,
        n,
        threshold,
        updated_a_threshold,
        2,
        BLOCK_SIZE
    )