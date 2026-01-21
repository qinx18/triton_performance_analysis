import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel computes the sum reduction for one iteration
    # Each block processes elements in chunks
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum for this block
    sum_val = 0.0
    
    # Process the 8 groups of 4 elements each (32 elements total)
    # Group 1: test(a) - sum elements 0,1,2,3
    group_offsets = block_start + offsets
    mask = group_offsets < 4
    if pid == 0:  # Only first block handles this
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Group 2: test(&a[4]) - sum elements 4,5,6,7
    group_offsets = block_start + offsets + 4
    mask = (group_offsets >= 4) & (group_offsets < 8)
    if pid == 0:
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Group 3: test(&a[8]) - sum elements 8,9,10,11
    group_offsets = block_start + offsets + 8
    mask = (group_offsets >= 8) & (group_offsets < 12)
    if pid == 0:
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Group 4: test(&a[12]) - sum elements 12,13,14,15
    group_offsets = block_start + offsets + 12
    mask = (group_offsets >= 12) & (group_offsets < 16)
    if pid == 0:
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Group 5: test(&a[16]) - sum elements 16,17,18,19
    group_offsets = block_start + offsets + 16
    mask = (group_offsets >= 16) & (group_offsets < 20)
    if pid == 0:
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Group 6: test(&a[20]) - sum elements 20,21,22,23
    group_offsets = block_start + offsets + 20
    mask = (group_offsets >= 20) & (group_offsets < 24)
    if pid == 0:
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Group 7: test(&a[24]) - sum elements 24,25,26,27
    group_offsets = block_start + offsets + 24
    mask = (group_offsets >= 24) & (group_offsets < 28)
    if pid == 0:
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Group 8: test(&a[28]) - sum elements 28,29,30,31
    group_offsets = block_start + offsets + 28
    mask = (group_offsets >= 28) & (group_offsets < 32)
    if pid == 0:
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Store result (only first block writes the final sum)
    if pid == 0:
        tl.store(result_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (1,)  # Single block to handle the reduction
    
    s31111_kernel[grid](
        a, result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result[0].item()