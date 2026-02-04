import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel computes a single sum value across all work
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Calculate sum for the 8 test() calls
    total_sum = 0.0
    
    # Each test() call sums 4 consecutive elements starting at different offsets
    offsets = [0, 4, 8, 12, 16, 20, 24, 28]
    
    for start_offset in offsets:
        # Sum 4 elements starting from start_offset
        test_sum = 0.0
        for i in range(4):
            idx = start_offset + i
            if idx < N:
                val = tl.load(a_ptr + idx)
                test_sum += val
        total_sum += test_sum
    
    # Store the final sum
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread
    grid = (1,)
    BLOCK_SIZE = 32
    
    s31111_kernel[grid](
        a, sum_result, N, BLOCK_SIZE
    )
    
    return sum_result[0].item()