import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    sum_val = 0.0
    
    # Process 8 groups of 4 elements each (indices 0,4,8,12,16,20,24,28)
    for group in range(8):
        base_idx = block_start + group * 4
        
        # Load 4 consecutive elements for this group
        group_offsets = base_idx + offsets
        mask = group_offsets < n_elements
        group_vals = tl.load(a_ptr + group_offsets, mask=mask, other=0.0)
        
        # Sum the first 4 elements (equivalent to test function)
        group_sum = 0.0
        for i in range(4):
            if i < BLOCK_SIZE:
                element_mask = (offsets == i) & mask
                element_val = tl.where(element_mask, group_vals, 0.0)
                group_sum += tl.sum(element_val)
        
        sum_val += group_sum
    
    # Store result (each block produces one sum)
    if pid == 0:
        tl.store(a_ptr + n_elements, sum_val)

def s31111_triton(a):
    n_elements = a.numel()
    
    # We need to process 32 elements total (8 groups * 4 elements each)
    # But we'll use the actual array size for bounds checking
    BLOCK_SIZE = 32
    
    # Allocate extra space for result
    result_tensor = torch.cat([a, torch.zeros(1, dtype=a.dtype, device=a.device)])
    
    # Launch kernel with single block since we're doing a global reduction
    grid = (1,)
    
    s31111_kernel[grid](
        result_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result_tensor[-1].item()