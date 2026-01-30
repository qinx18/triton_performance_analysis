import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(x_expanded_ptr, a_ptr, b_ptr, c_ptr, N):
    """Phase 1: Expand scalar x to array using sequential computation"""
    x_val = 0.0
    for i in range(N):
        # Compute x value for this iteration
        a_val = tl.load(a_ptr + (N - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        x_val = a_val + b_val * c_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, b_ptr, x_expanded_ptr, N, threshold, BLOCK_SIZE: tl.constexpr):
    """Phase 2: Parallel computation for first half using expanded x values"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = (idx < threshold) & (idx >= 0)
    
    x_vals = tl.load(x_expanded_ptr + idx, mask=mask, other=0.0)
    
    # Update arrays
    a_new = x_vals - 1.0
    b_new = x_vals
    
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, x_expanded_ptr, N, threshold, BLOCK_SIZE: tl.constexpr):
    """Phase 3: Parallel computation for second half using expanded x values"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < N
    
    x_vals = tl.load(x_expanded_ptr + idx, mask=mask, other=0.0)
    
    # Update arrays
    a_new = x_vals - 1.0
    b_new = x_vals
    
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)

def s281_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    threshold = N // 2
    
    # Create copy of original array a for reading
    a_copy = a.clone()
    
    # Create expanded scalar array
    x_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar x to array (sequential) using original values
    s281_expand_x_kernel[(1,)](
        x_expanded, a_copy, b, c, N
    )
    
    # Phase 2: Parallel computation for first half
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](
        a, b, x_expanded, N, threshold, BLOCK_SIZE
    )
    
    # Phase 3: Parallel computation for second half
    remaining = N - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](
            a, b, x_expanded, N, threshold, BLOCK_SIZE
        )