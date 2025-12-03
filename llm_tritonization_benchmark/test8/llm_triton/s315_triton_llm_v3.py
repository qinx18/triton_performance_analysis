import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_max_ptr, result_idx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find max element and its index using parallel reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first block
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find local max and indices
    local_max = tl.max(vals, axis=0)
    local_max_mask = vals == local_max
    local_indices = tl.where(local_max_mask, offsets, n_elements)
    local_min_idx = tl.min(local_indices, axis=0)
    
    current_max = local_max
    current_idx = local_min_idx
    
    # Process remaining blocks
    for block_start in range(BLOCK_SIZE, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        block_max = tl.max(vals, axis=0)
        
        # Update if we found a larger value
        if block_max > current_max:
            current_max = block_max
            block_max_mask = vals == block_max
            block_indices = tl.where(block_max_mask, current_offsets, n_elements)
            current_idx = tl.min(block_indices, axis=0)
    
    # Store results
    tl.store(result_max_ptr, current_max)
    tl.store(result_idx_ptr, current_idx)

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensors
    result_max = torch.zeros(1, dtype=a.dtype, device=a.device)
    result_idx = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s315_kernel[grid](
        a, result_max, result_idx, n_elements, BLOCK_SIZE
    )
    
    return result_max[0].item(), result_idx[0].item()