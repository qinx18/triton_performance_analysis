import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    
    if program_id >= 1:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Call test() function 8 times with stride 4: a[0], a[4], a[8], ..., a[28]
    for base_idx in range(8):
        start_idx = base_idx * 4
        
        # test() function: sum 4 consecutive elements starting at start_idx
        # Ensure we don't go out of bounds
        if start_idx + 3 < N:
            offsets = tl.arange(0, 4)
            indices = start_idx + offsets
            vals = tl.load(a_ptr + indices)
            local_sum = tl.sum(vals)
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