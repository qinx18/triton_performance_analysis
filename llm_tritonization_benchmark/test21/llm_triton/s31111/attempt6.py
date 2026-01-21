import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    # This kernel computes the sum reduction for the specific pattern
    # sum += test(a) + test(&a[4]) + ... + test(&a[28])
    # where test(ptr) sums 4 consecutive elements starting at ptr
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Compute sum for each test() call
    total_sum = 0.0
    
    # Each test() call processes 4 consecutive elements
    for start_offset in range(0, 32, 4):  # 0, 4, 8, 12, 16, 20, 24, 28
        # Load 4 elements starting at a[start_offset]
        offsets = tl.arange(0, 4) + start_offset
        vals = tl.load(a_ptr + offsets)
        # Sum these 4 elements
        test_sum = tl.sum(vals)
        total_sum += test_sum
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    # Allocate output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block
    grid = (1,)
    s31111_kernel[grid](a, sum_result, BLOCK_SIZE=32)
    
    return sum_result.item()