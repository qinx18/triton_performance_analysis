import triton
import triton.language as tl

@triton.jit
def s1113_kernel_phase1(a_ptr, a_copy_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n) & (indices <= threshold)
    
    a_val = tl.load(a_copy_ptr + threshold)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_val + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s1113_kernel_phase2(a_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + threshold + 1
    
    mask = indices < n
    
    a_val = tl.load(a_ptr + threshold)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_val + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Create read-only copy
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: i = 0 to threshold (inclusive)
    grid_phase1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel_phase1[grid_phase1](
        a, a_copy, b, n, threshold, BLOCK_SIZE
    )
    
    # Phase 2: i = threshold+1 to end
    if threshold + 1 < n:
        remaining = n - threshold - 1
        grid_phase2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel_phase2[grid_phase2](
            a, b, n, threshold, BLOCK_SIZE
        )