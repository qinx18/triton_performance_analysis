import triton
import triton.language as tl

@triton.jit
def s1113_kernel(a_ptr, b_ptr, n, constant_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute and store
    result = constant_val + b_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Save original value at threshold before any modifications
    orig_val = a[threshold].item()
    
    # Phase 1: i = 0 to threshold (inclusive) - uses original a[threshold]
    grid1 = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s1113_kernel[grid1](a, b, threshold + 1, orig_val, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold+1 to end - uses updated a[threshold]
    if threshold + 1 < n:
        remaining = n - threshold - 1
        a_phase2 = a[threshold + 1:]
        b_phase2 = b[threshold + 1:]
        
        # Get updated value at threshold
        updated_val = a[threshold].item()
        
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s1113_kernel[grid2](a_phase2, b_phase2, remaining, updated_val, BLOCK_SIZE=BLOCK_SIZE)