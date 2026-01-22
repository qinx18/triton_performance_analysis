import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single program computes the entire sum
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    sum_val = 0.0
    
    # Process each group of 4 elements: indices 0-3, 4-7, 8-11, 12-15, 16-19, 20-23, 24-27, 28-31
    for start_idx in range(8):
        base_idx = start_idx * 4
        
        # Load 4 elements starting at base_idx
        offsets = tl.arange(0, 4) + base_idx
        mask = offsets < N
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Sum the 4 elements (test function equivalent)
        group_sum = tl.sum(vals)
        sum_val += group_sum
    
    # Store the final sum
    tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, N, BLOCK_SIZE
    )
    
    return sum_result[0].item()