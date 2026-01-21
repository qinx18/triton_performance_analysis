import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize sum
    total_sum = 0.0
    
    # Process each of the 8 segments starting at positions 0, 4, 8, 12, 16, 20, 24, 28
    # Each segment sums 4 consecutive elements using the test() function logic
    for segment in range(8):
        segment_start = segment * 4
        segment_sum = 0.0
        
        # Sum 4 elements from this segment (implementing test() function)
        for i in range(4):
            idx = segment_start + i
            if idx < N:
                val = tl.load(a_ptr + idx)
                segment_sum += val
        
        total_sum += segment_sum
    
    # Store the final sum
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block since this is a reduction
    BLOCK_SIZE = 256
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()