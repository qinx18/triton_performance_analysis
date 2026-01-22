import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Load the 32 elements we need (8 groups of 4)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < 32
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Sum using vectorized operations with reshaping
        vals_reshaped = tl.reshape(vals, (8, 4))
        group_sums = tl.sum(vals_reshaped, axis=1)
        total_sum = tl.sum(group_sums)
        
        tl.store(sum_ptr, total_sum)

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