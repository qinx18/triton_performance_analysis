import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, block_results_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Check which elements are negative
    neg_mask = vals < 0.0
    valid_mask = mask & neg_mask
    
    # Get indices where condition is met
    valid_indices = tl.where(valid_mask, current_offsets, -1)
    
    # Find maximum valid index in this block
    max_idx = tl.max(valid_indices, axis=0)
    
    # Store the result for this block
    tl.store(block_results_ptr + pid, max_idx)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    if n_elements == 0:
        return -1
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    block_results = torch.full((num_blocks,), -1, dtype=torch.int32, device=a.device)
    
    grid = (num_blocks,)
    s331_kernel[grid](a, block_results, n_elements, BLOCK_SIZE)
    
    # Find maximum across all blocks
    valid_results = block_results[block_results >= 0]
    if len(valid_results) > 0:
        j = torch.max(valid_results).item()
    else:
        j = -1
    
    return j