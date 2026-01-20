import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to handle scalar expansion
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    x_val = 0.0
    for i in range(n):
        # Load values for position i
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # Compute x value
        x_val = a_val + b_val * c_val
        
        # Store expanded x value
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, x_expanded_ptr, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < threshold
    
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    a_new = x_vals - 1.0
    
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, x_expanded_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    x_vals = tl.load(x_expanded_ptr + indices, mask=mask)
    
    a_new = x_vals - 1.0
    
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    
    # Step 1: Expand scalar x to array
    grid_expand = (1,)
    s281_expand_x_kernel[grid_expand](a, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 2: Phase 1 - parallel computation for indices 0 to threshold-1
    a_copy = a.clone()
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, x_expanded, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 3: Phase 2 - parallel computation for indices threshold to n-1
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](a, b, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)