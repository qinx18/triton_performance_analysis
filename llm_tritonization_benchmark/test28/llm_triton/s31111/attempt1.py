import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel computes sum of test() calls on specific array positions
    # Each test() sums 4 consecutive elements starting from given position
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Compute sum for each test() call
    total_sum = 0.0
    
    # Positions for test() calls: 0, 4, 8, 12, 16, 20, 24, 28
    for base_idx in range(0, 32, 4):
        test_sum = 0.0
        # Each test() sums 4 consecutive elements
        for i in range(4):
            idx = base_idx + i
            if idx < N:
                val = tl.load(a_ptr + idx)
                test_sum += val
        total_sum += test_sum
    
    tl.store(result_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread since this is a reduction
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()