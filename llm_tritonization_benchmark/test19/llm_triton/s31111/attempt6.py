import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, BLOCK_SIZE: tl.constexpr):
    # Calculate sum of elements at specific offsets
    # Each offset group: a[i], a[i+4], a[i+8], ..., a[i+28] where i is the base offset
    base_offsets = tl.arange(0, 8) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]
    
    sum_val = 0.0
    
    # For each base offset, sum 4 consecutive elements (test function)
    for i in range(8):
        base = base_offsets[i]
        # Sum 4 consecutive elements starting from base
        for j in range(4):
            offset = base + j
            val = tl.load(a_ptr + offset)
            sum_val += val
    
    # Store the final sum
    tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor for the sum
    sum_result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single thread since we're computing one sum
    grid = (1,)
    s31111_kernel[grid](a, sum_result, BLOCK_SIZE=32)
    
    return sum_result.item()