import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one iteration of sum computation
    program_id = tl.program_id(0)
    
    if program_id >= 1:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Manually unroll the 8 test() calls with stride 4
    for base_idx in range(8):
        start_idx = base_idx * 4
        
        # Sum 4 consecutive elements starting from start_idx
        local_sum = 0.0
        for i in range(4):
            idx = start_idx + i
            if idx < N:
                val = tl.load(a_ptr + idx)
                local_sum += val
        
        sum_val += local_sum
    
    # Store the result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    # Create output tensor for the sum result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s31111_kernel[grid](
        a, result, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()