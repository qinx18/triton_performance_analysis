import triton
import triton.language as tl
import torch

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n):
    # Single thread processes all elements sequentially
    x_val = 0.0
    for i in range(n):
        # Load values
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # Compute x
        x_val = a_val + b_val * c_val
        
        # Store expanded x value
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, x_expanded_ptr, 
                       start_idx, n_elements, BLOCK_SIZE: tl.constexpr, n_total):
    block_id = tl.program_id(0)
    offsets = start_idx + block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (start_idx + n_elements)
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Store results
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, 
                       start_idx, n_elements, BLOCK_SIZE: tl.constexpr, n_total):
    block_id = tl.program_id(0)
    offsets = start_idx + block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (start_idx + n_elements)
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Store results
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    
    # Create expanded x array
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Clone array for reads
    a_copy = a.clone()
    
    # Step 1: Expand scalar x using sequential kernel
    grid = (1,)
    s281_expand_x_kernel[grid](
        a_copy, b, c, x_expanded, n
    )
    
    # Step 2: Phase 1 - parallel execution for i = 0 to threshold-1
    BLOCK_SIZE = 256
    if threshold > 0:
        grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
        s281_phase1_kernel[grid1](
            a, a_copy, b, c, x_expanded, 0, threshold, BLOCK_SIZE, n
        )
    
    # Step 3: Phase 2 - parallel execution for i = threshold to n-1
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](
            a, b, c, x_expanded, threshold, remaining, BLOCK_SIZE, n
        )