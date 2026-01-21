import triton
import triton.language as tl
import torch

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    if tl.program_id(0) == 0:
        for i in range(n):
            a_val = tl.load(a_ptr + (n - 1 - i))
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            x_val = a_val + b_val * c_val
            tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < threshold
    
    # Load expanded x values and compute results
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    # Store results
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load expanded x values and compute results
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    # Store results
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create expanded x array
    x_expanded = torch.zeros_like(a)
    
    # Step 1: Expand scalar x to array (sequential)
    grid_expand = (1,)
    s281_expand_x_kernel[grid_expand](a, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 2: Phase 1 - Process first half (parallel)
    if threshold > 0:
        a_copy = a.clone()  # Clone for reading original values
        grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
        s281_phase1_kernel[grid1](a, a_copy, b, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 3: Phase 2 - Process second half (parallel)
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](a, b, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)