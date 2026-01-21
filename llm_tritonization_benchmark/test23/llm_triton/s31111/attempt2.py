import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    sum_val = 0.0
    
    # Group 1: sum elements 0,1,2,3
    mask = offsets < 4
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Group 2: sum elements 4,5,6,7
    group_offsets = offsets + 4
    mask = offsets < 4
    vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Group 3: sum elements 8,9,10,11
    group_offsets = offsets + 8
    mask = offsets < 4
    vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Group 4: sum elements 12,13,14,15
    group_offsets = offsets + 12
    mask = offsets < 4
    vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Group 5: sum elements 16,17,18,19
    group_offsets = offsets + 16
    mask = offsets < 4
    vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Group 6: sum elements 20,21,22,23
    group_offsets = offsets + 20
    mask = offsets < 4
    vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Group 7: sum elements 24,25,26,27
    group_offsets = offsets + 24
    mask = offsets < 4
    vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Group 8: sum elements 28,29,30,31
    group_offsets = offsets + 28
    mask = offsets < 4
    vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
    sum_val += tl.sum(vals)
    
    # Store result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result[0].item()