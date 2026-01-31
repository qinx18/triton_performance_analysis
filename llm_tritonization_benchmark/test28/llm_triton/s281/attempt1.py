import torch
import triton
import triton.language as tl

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially for scalar expansion
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    x_val = 0.0
    for i in range(n_elements):
        # Load values
        a_val = tl.load(a_ptr + (n_elements - i - 1))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # Compute x
        x_val = a_val + b_val * c_val
        
        # Store expanded x value
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, x_expanded_ptr, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < threshold
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Update arrays - phase 1
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, threshold, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = threshold + pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Update arrays - phase 2
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = (n + 1) // 2
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros_like(a)
    
    # Clone array for crossing threshold pattern
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar x
    grid = (1,)
    s281_expand_x_kernel[grid](
        a_copy, b, c, x_expanded, n, BLOCK_SIZE
    )
    
    # Phase 2: Update arrays phase 1 (i = 0 to threshold-1)
    grid = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid](
        a, a_copy, b, c, x_expanded, threshold, BLOCK_SIZE
    )
    
    # Phase 3: Update arrays phase 2 (i = threshold to n-1)
    remaining = n - threshold
    if remaining > 0:
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid](
            a, b, c, x_expanded, threshold, n, BLOCK_SIZE
        )