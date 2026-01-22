import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one iteration independently
    pid = tl.program_id(0)
    
    # Initialize sum for this program
    sum_val = 0.0
    
    # Process 8 groups of 4 elements each (starting at indices 0, 4, 8, 12, 16, 20, 24, 28)
    for group_idx in range(8):
        start_idx = group_idx * 4
        
        # Load 4 consecutive elements starting at start_idx
        offsets = start_idx + tl.arange(0, 4)
        mask = offsets < N
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Sum the 4 elements (equivalent to test() function)
        group_sum = tl.sum(vals)
        sum_val += group_sum
    
    # Store the result
    tl.store(sum_ptr + pid, sum_val)

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