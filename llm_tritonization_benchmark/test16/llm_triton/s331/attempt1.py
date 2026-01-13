import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize local result to -1
    local_j = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    
    for start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check condition: a[i] < 0
        condition = vals < 0.0
        
        # Update j for elements that satisfy condition
        indices = current_offsets
        local_j = tl.where(condition & mask, indices, local_j)
    
    # Find the maximum valid index across the block
    final_j = tl.max(local_j, axis=0)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, final_j)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.tensor(-1, dtype=torch.int32, device=a.device)
    
    # Launch kernel with single block to ensure correct reduction
    grid = (1,)
    s331_kernel[grid](a, result, n_elements, BLOCK_SIZE)
    
    return result.item()