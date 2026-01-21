import torch
import triton
import triton.language as tl

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n // 2
    
    # Load data for phase 1
    reverse_offsets = n - 1 - offsets
    a_vals = tl.load(a_copy_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x values for this block
    x_vals = a_vals + b_vals * c_vals
    
    # Store expanded x values
    tl.store(x_expanded_ptr + offsets, x_vals, mask=mask)
    
    # Store results
    a_results = x_vals - 1.0
    tl.store(a_ptr + offsets, a_results, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    threshold = n // 2
    block_start = threshold + pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load data for phase 2
    reverse_offsets = n - 1 - offsets
    a_vals = tl.load(a_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x values
    x_vals = a_vals + b_vals * c_vals
    
    # Store expanded x values
    tl.store(x_expanded_ptr + offsets, x_vals, mask=mask)
    
    # Store results
    a_results = x_vals - 1.0
    tl.store(a_ptr + offsets, a_results, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create expanded x array
    x_expanded = torch.zeros_like(a)
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    # Phase 1: Process first half
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](
        a, a_copy, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Process second half
    remaining = n - threshold
    if remaining > 0:
        grid2 = (triton.cdiv(remaining, BLOCK_SIZE),)
        s281_phase2_kernel[grid2](
            a, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Store final x value
    x.fill_(x_expanded[-1].item())