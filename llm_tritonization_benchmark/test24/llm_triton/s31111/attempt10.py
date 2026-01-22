import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # We need to call test() on 8 different starting positions: 0, 4, 8, 12, 16, 20, 24, 28
        # Each test() call sums 4 consecutive elements
        total_sum = 0.0
        
        # Load all 32 elements at once
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < 32
        vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Sum elements [0:4] (test(a))
        total_sum += vals[0] + vals[1] + vals[2] + vals[3]
        
        # Sum elements [4:8] (test(&a[4]))
        total_sum += vals[4] + vals[5] + vals[6] + vals[7]
        
        # Sum elements [8:12] (test(&a[8]))
        total_sum += vals[8] + vals[9] + vals[10] + vals[11]
        
        # Sum elements [12:16] (test(&a[12]))
        total_sum += vals[12] + vals[13] + vals[14] + vals[15]
        
        # Sum elements [16:20] (test(&a[16]))
        total_sum += vals[16] + vals[17] + vals[18] + vals[19]
        
        # Sum elements [20:24] (test(&a[20]))
        total_sum += vals[20] + vals[21] + vals[22] + vals[23]
        
        # Sum elements [24:28] (test(&a[24]))
        total_sum += vals[24] + vals[25] + vals[26] + vals[27]
        
        # Sum elements [28:32] (test(&a[28]))
        total_sum += vals[28] + vals[29] + vals[30] + vals[31]
        
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