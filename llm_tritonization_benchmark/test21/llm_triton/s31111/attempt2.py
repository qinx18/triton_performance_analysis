import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize sum
    sum_val = 0.0
    
    # Sum 8 groups of 4 elements each: indices [0:4], [4:8], [8:12], [12:16], [16:20], [20:24], [24:28], [28:32]
    for group_idx in range(8):
        base_idx = group_idx * 4
        # Load 4 consecutive elements
        offsets = tl.arange(0, 4) + base_idx
        mask = offsets < n
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        # Sum the 4 elements (equivalent to test() function)
        group_sum = tl.sum(vals)
        sum_val += group_sum
    
    # Store the result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n = a.shape[0]
    
    # Ensure we have at least 32 elements
    if n < 32:
        a_padded = torch.zeros(32, dtype=a.dtype, device=a.device)
        a_padded[:n] = a
        a = a_padded
        n = 32
    
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