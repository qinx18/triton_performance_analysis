import torch
import triton
import triton.language as tl

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
def s281_phase1_kernel(a_copy_ptr, b_ptr, c_ptr, x_expanded_ptr, a_ptr, b_out_ptr, 
                       n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = (current_offsets < threshold) & (current_offsets >= 0)
    
    # Load x values
    x_vals = tl.load(x_expanded_ptr + current_offsets, mask=mask)
    
    # Compute and store results
    a_vals = x_vals - 1.0
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    tl.store(b_out_ptr + current_offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, b_out_ptr,
                       n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = (current_offsets < n) & (current_offsets >= threshold)
    
    # Load x values
    x_vals = tl.load(x_expanded_ptr + current_offsets, mask=mask)
    
    # Compute and store results
    a_vals = x_vals - 1.0
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    tl.store(b_out_ptr + current_offsets, x_vals, mask=mask)

def s281_triton(a, b, c):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create arrays for scalar expansion
    x_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    a_copy = a.clone()
    
    # Expand scalar x
    s281_expand_x_kernel[(1,)](a, b, c, x_expanded, n)
    
    # Phase 1: indices 0 to threshold-1
    grid_size_1 = triton.cdiv(threshold, BLOCK_SIZE)
    if grid_size_1 > 0:
        s281_phase1_kernel[(grid_size_1,)](
            a_copy, b, c, x_expanded, a, b, 
            n, threshold, BLOCK_SIZE
        )
    
    # Phase 2: indices threshold to n-1
    remaining = n - threshold
    grid_size_2 = triton.cdiv(remaining, BLOCK_SIZE)
    if grid_size_2 > 0:
        s281_phase2_kernel[(grid_size_2,)](
            a, b, c, x_expanded, b,
            n, threshold, BLOCK_SIZE
        )