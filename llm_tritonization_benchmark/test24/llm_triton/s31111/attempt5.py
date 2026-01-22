import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    sum_val = 0.0
    
    # Process 8 groups of 4 elements each: a[0:3], a[4:7], ..., a[28:31]
    offsets = tl.arange(0, 4)
    
    for group in range(8):
        start_idx = group * 4
        
        # Load 4 consecutive elements
        current_offsets = start_idx + offsets
        mask = current_offsets < N
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Sum the 4 elements (equivalent to test() function)
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