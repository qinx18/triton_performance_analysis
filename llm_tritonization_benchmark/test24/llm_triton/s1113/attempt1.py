import triton
import triton.language as tl

@triton.jit
def s1113_kernel(a_ptr, a_copy_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Phase 1: i = 0 to threshold (uses original value)
    orig_value = tl.load(a_copy_ptr + threshold)
    
    for block_start in range(0, threshold + 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets <= threshold
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        result = orig_value + b_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    if threshold + 1 < n:
        updated_value = tl.load(a_ptr + threshold)
        
        for block_start in range(threshold + 1, n, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n
            
            b_vals = tl.load(b_ptr + current_offsets, mask=mask)
            result = updated_value + b_vals
            tl.store(a_ptr + current_offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Create read-only copy before launching kernel
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s1113_kernel[grid](
        a, a_copy, b, n, threshold, BLOCK_SIZE=BLOCK_SIZE
    )