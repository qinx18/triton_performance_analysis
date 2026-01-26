import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Since we only have one block processing the entire computation
    if pid != 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Process 8 groups of 4 elements each (indices 0, 4, 8, 12, 16, 20, 24, 28)
    for group in range(8):
        start_idx = group * 4
        group_sum = 0.0
        
        # Sum 4 elements starting from start_idx
        for i in range(4):
            idx = start_idx + i
            if idx < N:
                val = tl.load(a_ptr + idx)
                group_sum += val
        
        sum_val += group_sum
    
    # Store result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    BLOCK_SIZE = 128
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, N, BLOCK_SIZE
    )
    
    return result.item()