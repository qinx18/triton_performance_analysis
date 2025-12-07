import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program processes one group of 32 elements (8 groups of 4)
    pid = tl.program_id(0)
    block_start = pid * 32
    
    # Check if this block is within bounds
    if block_start >= n_elements:
        return
    
    # Initialize sum for this block
    sum_val = 0.0
    
    # Process 8 groups of 4 elements each (offsets 0, 4, 8, 12, 16, 20, 24, 28)
    for group_idx in range(8):
        group_offset = block_start + group_idx * 4
        
        # Load 4 elements for this group
        offsets = tl.arange(0, 4)
        element_offsets = group_offset + offsets
        mask = element_offsets < n_elements
        
        if group_offset < n_elements:
            elements = tl.load(a_ptr + element_offsets, mask=mask, other=0.0)
            # Sum the 4 elements (equivalent to test() function)
            group_sum = tl.sum(elements)
            sum_val += group_sum
    
    # Store the result
    tl.store(sum_ptr + pid, sum_val)

def s31111_triton(a):
    n_elements = a.shape[0]
    
    # Calculate number of blocks needed (each block processes 32 elements)
    n_blocks = triton.cdiv(n_elements, 32)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 128
    
    # Launch kernel
    grid = (n_blocks,)
    s31111_kernel[grid](
        a, partial_sums, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final sum
    result = torch.sum(partial_sums)
    return result