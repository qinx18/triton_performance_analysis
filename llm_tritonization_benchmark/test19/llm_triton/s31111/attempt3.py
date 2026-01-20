import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread to compute the sum
    thread_id = tl.program_id(0)
    
    if thread_id == 0:
        total_sum = 0.0
        
        # Manually unroll the 8 calls to test function
        # Each test() call sums 4 consecutive elements starting at specific offsets
        offsets = [0, 4, 8, 12, 16, 20, 24, 28]
        
        for start_idx in offsets:
            # Sum 4 elements starting at start_idx (equivalent to test() function)
            group_sum = 0.0
            for i in range(4):
                idx = start_idx + i
                if idx < N:
                    val = tl.load(a_ptr + idx)
                    group_sum += val
            total_sum += group_sum
        
        # Store the result
        tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result[0].item()