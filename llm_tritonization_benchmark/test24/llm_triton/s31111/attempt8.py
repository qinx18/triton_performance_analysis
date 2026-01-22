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
        
        # Sum each group of 4 elements using vectorized operations
        sum_total = 0.0
        
        # Group 0: elements 0-3
        group_sum = vals[0] + vals[1] + vals[2] + vals[3]
        sum_total += group_sum
        
        # Group 1: elements 4-7
        group_sum = vals[4] + vals[5] + vals[6] + vals[7]
        sum_total += group_sum
        
        # Group 2: elements 8-11
        group_sum = vals[8] + vals[9] + vals[10] + vals[11]
        sum_total += group_sum
        
        # Group 3: elements 12-15
        group_sum = vals[12] + vals[13] + vals[14] + vals[15]
        sum_total += group_sum
        
        # Group 4: elements 16-19
        group_sum = vals[16] + vals[17] + vals[18] + vals[19]
        sum_total += group_sum
        
        # Group 5: elements 20-23
        group_sum = vals[20] + vals[21] + vals[22] + vals[23]
        sum_total += group_sum
        
        # Group 6: elements 24-27
        group_sum = vals[24] + vals[25] + vals[26] + vals[27]
        sum_total += group_sum
        
        # Group 7: elements 28-31
        group_sum = vals[28] + vals[29] + vals[30] + vals[31]
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