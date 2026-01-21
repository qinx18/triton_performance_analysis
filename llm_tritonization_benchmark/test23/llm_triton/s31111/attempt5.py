import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, result_ptr, BLOCK_SIZE: tl.constexpr):
    # Initialize sum
    sum_val = 0.0
    
    # Create offsets for loading 4 elements at a time
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sum elements for each group of 4
    for start_idx in range(0, 32, 4):
        # Load 4 elements starting from start_idx
        load_offsets = start_idx + tl.arange(0, 4)
        vals = tl.load(a_ptr + load_offsets)
        sum_val += tl.sum(vals)
    
    # Store result
    tl.store(result_ptr, sum_val)

def s31111_triton(a):
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = (1,)
    
    s31111_kernel[grid](
        a, result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result[0].item()