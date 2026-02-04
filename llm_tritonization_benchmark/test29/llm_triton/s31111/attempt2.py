import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    total_sum = 0.0
    
    # Manually unroll the 8 test() calls instead of using a loop
    # test(a) - sum elements 0,1,2,3
    test_sum = 0.0
    for i in range(4):
        idx = 0 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    # test(&a[4]) - sum elements 4,5,6,7
    test_sum = 0.0
    for i in range(4):
        idx = 4 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    # test(&a[8]) - sum elements 8,9,10,11
    test_sum = 0.0
    for i in range(4):
        idx = 8 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    # test(&a[12]) - sum elements 12,13,14,15
    test_sum = 0.0
    for i in range(4):
        idx = 12 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    # test(&a[16]) - sum elements 16,17,18,19
    test_sum = 0.0
    for i in range(4):
        idx = 16 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    # test(&a[20]) - sum elements 20,21,22,23
    test_sum = 0.0
    for i in range(4):
        idx = 20 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    # test(&a[24]) - sum elements 24,25,26,27
    test_sum = 0.0
    for i in range(4):
        idx = 24 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    # test(&a[28]) - sum elements 28,29,30,31
    test_sum = 0.0
    for i in range(4):
        idx = 28 + i
        if idx < N:
            val = tl.load(a_ptr + idx)
            test_sum += val
    total_sum += test_sum
    
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    BLOCK_SIZE = 32
    
    s31111_kernel[grid](
        a, sum_result, N, BLOCK_SIZE
    )
    
    return sum_result[0].item()