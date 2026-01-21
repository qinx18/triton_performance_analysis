import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    if tl.program_id(axis=0) == 0:
        x_val = 0.0
        for i in range(n):
            a_val = tl.load(a_ptr + (n - 1 - i))
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            x_val = a_val + b_val * c_val
            tl.store(x_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, b_ptr, x_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < threshold
    
    # Load x values
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    
    # Store results
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, x_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load x values
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    
    # Store results
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar x to array (sequential)
    grid1 = (1,)
    s281_expand_x_kernel[grid1](a, b, c, x, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Process first half in parallel
    grid2 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid2](a, b, x, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 3: Process second half in parallel
    grid3 = (triton.cdiv(n - threshold, BLOCK_SIZE),)
    s281_phase2_kernel[grid3](a, b, x, n, threshold, BLOCK_SIZE=BLOCK_SIZE)