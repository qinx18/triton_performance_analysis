import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    sum_val = 0.0
    
    # For each starting position, sum 4 consecutive elements
    start_positions = [0, 4, 8, 12, 16, 20, 24, 28]
    
    for start_idx in start_positions:
        # Sum 4 elements starting at start_idx
        for i in range(4):
            offset = start_idx + i
            if offset < n:
                val = tl.load(a_ptr + offset)
                sum_val += val
    
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    n = a.shape[0]
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s31111_kernel[grid](a, result, n, BLOCK_SIZE=32)
    
    return result[0].item()