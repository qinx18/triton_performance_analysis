import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one iteration
    pid = tl.program_id(0)
    
    if pid >= 1:
        return
    
    # Calculate sum for specific indices: 0, 4, 8, 12, 16, 20, 24, 28
    # Each of these indices is the start of a 4-element block to sum
    total_sum = 0.0
    
    # Process each starting index
    for start_idx in range(8):  # 8 blocks starting at 0, 4, 8, 12, 16, 20, 24, 28
        block_start = start_idx * 4
        
        # Sum 4 elements starting from block_start
        block_sum = 0.0
        for i in range(4):
            idx = block_start + i
            if idx < N:
                val = tl.load(a_ptr + idx)
                block_sum += val
        
        total_sum += block_sum
    
    # Store the total sum
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result