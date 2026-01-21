import triton
import triton.language as tl

@triton.jit
def s1113_kernel(a_ptr, a_copy_ptr, b_ptr, n, threshold, phase, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    if phase == 1:
        # Phase 1: i = 0 to threshold (uses original a[threshold])
        mask = (offsets < n) & (offsets <= threshold)
        
        # Load the original value at threshold from the copy
        threshold_val = tl.load(a_copy_ptr + threshold)
        
        # Load b values
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        
        # Compute and store
        result = threshold_val + b_vals
        tl.store(a_ptr + offsets, result, mask=mask)
        
    else:  # phase == 2
        # Phase 2: i = threshold+1 to end (uses updated a[threshold])
        start_idx = threshold + 1
        mask = (offsets < n) & (offsets >= start_idx)
        
        # Load the updated value at threshold from the original array
        threshold_val = tl.load(a_ptr + threshold)
        
        # Load b values
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        
        # Compute and store
        result = threshold_val + b_vals
        tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create read-only copy of a before any modifications
    a_copy = a.clone()
    
    # Phase 1: i = 0 to threshold (inclusive) - uses original a[threshold]
    grid1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel[grid1](a, a_copy, b, n, threshold, 1, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold+1 to end - uses updated a[threshold]
    if threshold + 1 < n:
        grid2 = (triton.cdiv(n - threshold - 1, BLOCK_SIZE),)
        s1113_kernel[grid2](a, a_copy, b, n, threshold, 2, BLOCK_SIZE=BLOCK_SIZE)