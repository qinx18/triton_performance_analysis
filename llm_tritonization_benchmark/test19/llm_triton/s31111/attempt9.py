import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread computes the entire sum
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    sum_val = 0.0
    
    # For each of the 8 starting positions: 0, 4, 8, 12, 16, 20, 24, 28
    # Sum 4 consecutive elements starting from each position
    for start_idx in range(8):
        base_offset = start_idx * 4
        
        # Load and sum 4 elements starting at base_offset
        for i in range(4):
            offset = base_offset + i
            if offset < n:
                val = tl.load(a_ptr + offset)
                sum_val += val
    
    # Store the final result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n = a.shape[0]
    
    # Create output tensor for the sum result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread
    grid = (1,)
    s31111_kernel[grid](a, result, n, BLOCK_SIZE=32)
    
    return result[0].item()