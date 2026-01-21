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
    
    # Manually unroll the 8 test() calls with correct offsets
    # test(a) - sum a[0] to a[3]
    offsets = tl.arange(0, 4)
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    # test(&a[4]) - sum a[4] to a[7]
    offsets = tl.arange(0, 4) + 4
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    # test(&a[8]) - sum a[8] to a[11]
    offsets = tl.arange(0, 4) + 8
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    # test(&a[12]) - sum a[12] to a[15]
    offsets = tl.arange(0, 4) + 12
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    # test(&a[16]) - sum a[16] to a[19]
    offsets = tl.arange(0, 4) + 16
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    # test(&a[20]) - sum a[20] to a[23]
    offsets = tl.arange(0, 4) + 20
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    # test(&a[24]) - sum a[24] to a[27]
    offsets = tl.arange(0, 4) + 24
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    # test(&a[28]) - sum a[28] to a[31]
    offsets = tl.arange(0, 4) + 28
    vals = tl.load(a_ptr + offsets)
    total_sum += tl.sum(vals)
    
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s31111_kernel[grid](a, sum_result, BLOCK_SIZE=32)
    
    return sum_result.item()