import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize sum
    sum_val = 0.0
    
    # Sum 8 groups of 4 elements each starting at indices 0, 4, 8, 12, 16, 20, 24, 28
    for group_idx in range(8):
        base_idx = group_idx * 4
        # Load 4 consecutive elements for this group
        offsets = tl.arange(0, 4)
        group_offsets = base_idx + offsets
        mask = group_offsets < n
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        # Sum the 4 elements from each position
        for i in range(4):
            sum_val += vals[i]
    
    # Store the result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n = a.shape[0]
    
    # Create output tensor for single result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    grid = (1,)
    BLOCK_SIZE = 32
    
    s31111_kernel[grid](
        a, result, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()