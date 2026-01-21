import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize sum
    total_sum = 0.0
    
    # Sum 8 groups of 4 elements each
    for start_idx in range(0, 32, 4):
        group_sum = 0.0
        for i in range(4):
            idx = start_idx + i
            if idx < n_elements:
                val = tl.load(a_ptr + idx)
                group_sum += val
        total_sum += group_sum
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_result, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result.item()