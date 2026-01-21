import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize sum
    total_sum = 0.0
    
    # Load all elements we need (a[0] through a[31])
    offsets = tl.arange(0, 32)
    vals = tl.load(a_ptr + offsets)
    
    # Sum groups of 4 elements each using individual element access
    # test(a) - sum a[0:4]
    group_sum = vals[0] + vals[1] + vals[2] + vals[3]
    total_sum += group_sum
    
    # test(&a[4]) - sum a[4:8]  
    group_sum = vals[4] + vals[5] + vals[6] + vals[7]
    total_sum += group_sum
    
    # test(&a[8]) - sum a[8:12]
    group_sum = vals[8] + vals[9] + vals[10] + vals[11]
    total_sum += group_sum
    
    # test(&a[12]) - sum a[12:16]
    group_sum = vals[12] + vals[13] + vals[14] + vals[15]
    total_sum += group_sum
    
    # test(&a[16]) - sum a[16:20]
    group_sum = vals[16] + vals[17] + vals[18] + vals[19]
    total_sum += group_sum
    
    # test(&a[20]) - sum a[20:24]
    group_sum = vals[20] + vals[21] + vals[22] + vals[23]
    total_sum += group_sum
    
    # test(&a[24]) - sum a[24:28]
    group_sum = vals[24] + vals[25] + vals[26] + vals[27]
    total_sum += group_sum
    
    # test(&a[28]) - sum a[28:32]
    group_sum = vals[28] + vals[29] + vals[30] + vals[31]
    total_sum += group_sum
    
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s31111_kernel[grid](a, sum_result, BLOCK_SIZE=32)
    
    return sum_result.item()