import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate the sum by loading and summing 8 groups of 4 elements each
    sum_val = 0.0
    
    # Process each group of 4 elements starting at positions 0, 4, 8, ..., 28
    for start_idx in range(0, 32, 4):
        # Load 4 elements starting at start_idx
        group_offsets = start_idx + tl.arange(0, 4)
        mask = group_offsets < n_elements
        vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        group_sum = tl.sum(vals)
        sum_val += group_sum
    
    # Store the result (only first thread stores the final sum)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we're computing a single sum
    BLOCK_SIZE = 128
    grid = (1,)
    
    s31111_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output