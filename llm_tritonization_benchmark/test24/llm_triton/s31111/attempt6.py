import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Load the 32 elements we need (8 groups of 4)
        offsets = tl.arange(0, 32)
        mask = offsets < 32
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Sum each group of 4 elements
        sum_total = 0.0
        for i in range(8):
            group_start = i * 4
            group_sum = vals[group_start] + vals[group_start + 1] + vals[group_start + 2] + vals[group_start + 3]
            sum_total += group_sum
        
        tl.store(sum_ptr, sum_total)

def s31111_triton(a):
    # Allocate output tensor for sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    BLOCK_SIZE = 32
    
    s31111_kernel[grid](
        a, sum_result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()