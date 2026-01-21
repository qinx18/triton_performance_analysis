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
    
    # Sum groups of 4 elements each using slicing
    # test(a) - sum a[0:4]
    total_sum += tl.sum(vals[0:4])
    
    # test(&a[4]) - sum a[4:8]  
    total_sum += tl.sum(vals[4:8])
    
    # test(&a[8]) - sum a[8:12]
    total_sum += tl.sum(vals[8:12])
    
    # test(&a[12]) - sum a[12:16]
    total_sum += tl.sum(vals[12:16])
    
    # test(&a[16]) - sum a[16:20]
    total_sum += tl.sum(vals[16:20])
    
    # test(&a[20]) - sum a[20:24]
    total_sum += tl.sum(vals[20:24])
    
    # test(&a[24]) - sum a[24:28]
    total_sum += tl.sum(vals[24:28])
    
    # test(&a[28]) - sum a[28:32]
    total_sum += tl.sum(vals[28:32])
    
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s31111_kernel[grid](a, sum_result, BLOCK_SIZE=32)
    
    return sum_result.item()