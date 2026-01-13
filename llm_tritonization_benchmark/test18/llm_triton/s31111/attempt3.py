import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single block computes the entire sum
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Load all 32 elements at once (8 groups of 4)
    offsets = tl.arange(0, 32)
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Use tl.sum to sum groups of 4 elements
    # Group 1: elements 0-3
    group1_offsets = tl.arange(0, 4)
    group1_vals = tl.load(a_ptr + group1_offsets, mask=group1_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group1_vals)
    
    # Group 2: elements 4-7  
    group2_offsets = tl.arange(4, 8)
    group2_vals = tl.load(a_ptr + group2_offsets, mask=group2_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group2_vals)
    
    # Group 3: elements 8-11
    group3_offsets = tl.arange(8, 12)
    group3_vals = tl.load(a_ptr + group3_offsets, mask=group3_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group3_vals)
    
    # Group 4: elements 12-15
    group4_offsets = tl.arange(12, 16)
    group4_vals = tl.load(a_ptr + group4_offsets, mask=group4_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group4_vals)
    
    # Group 5: elements 16-19
    group5_offsets = tl.arange(16, 20)
    group5_vals = tl.load(a_ptr + group5_offsets, mask=group5_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group5_vals)
    
    # Group 6: elements 20-23
    group6_offsets = tl.arange(20, 24)
    group6_vals = tl.load(a_ptr + group6_offsets, mask=group6_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group6_vals)
    
    # Group 7: elements 24-27
    group7_offsets = tl.arange(24, 28)
    group7_vals = tl.load(a_ptr + group7_offsets, mask=group7_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group7_vals)
    
    # Group 8: elements 28-31
    group8_offsets = tl.arange(28, 32)
    group8_vals = tl.load(a_ptr + group8_offsets, mask=group8_offsets < n_elements, other=0.0)
    sum_val += tl.sum(group8_vals)
    
    # Store result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 32
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch single block
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a