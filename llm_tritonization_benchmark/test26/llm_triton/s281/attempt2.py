import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n):
    # Single thread processes all elements sequentially
    x_val = 0.0
    for i in range(n):
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        x_val = a_val + b_val * c_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, b_ptr, x_expanded_ptr, 
                       threshold, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < threshold
    
    x_vals = tl.load(x_expanded_ptr + idx, mask=mask)
    
    tl.store(a_ptr + idx, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + idx, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, x_expanded_ptr, 
                       threshold, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = threshold + block_start + offsets
    
    mask = idx < n
    
    x_vals = tl.load(x_expanded_ptr + idx, mask=mask)
    
    tl.store(a_ptr + idx, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + idx, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Clone array for Phase 1 reads
    a_copy = a.clone()
    
    # Create expanded scalar array
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Expand scalar x sequentially
    expand_x_kernel[(1,)](a_copy, b, c, x_expanded, n)
    
    # Phase 1: i = 0 to threshold-1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid](a, b, x_expanded, threshold, BLOCK_SIZE)
    
    # Phase 2: i = threshold to n-1
    remaining = n - threshold
    if remaining > 0:
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid](a, b, x_expanded, threshold, n, BLOCK_SIZE)