import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate sum of elements at specific positions
    # Each position sums 4 consecutive elements starting from base indices
    base_indices = tl.arange(0, 8) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]
    
    total_sum = 0.0
    
    for i in range(8):
        base_idx = i * 4
        if base_idx + 3 < N:
            # Load 4 consecutive elements starting from base_idx
            offsets = base_idx + tl.arange(0, 4)
            mask = offsets < N
            vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
            # Sum the 4 elements
            partial_sum = tl.sum(vals)
            total_sum += partial_sum
    
    # Store the result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    N = a.shape[0]
    
    # Output tensor for the sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 32
    
    # Launch kernel with a single block since we're computing one sum
    s31111_kernel[(1,)](
        a, sum_result, N, BLOCK_SIZE
    )
    
    return sum_result.item()