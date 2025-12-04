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
    
    # Find the highest index in this block that is negative
    max_idx = -1
    
    # Convert mask to integer for iteration
    valid_indices = tl.where(valid_mask, current_offsets, -1)
    
    # Find maximum valid index using reduction-like approach
    for i in tl.static_range(BLOCK_SIZE):
        candidate_idx = tl.load(tl.pointer_type(tl.int32)(&valid_indices) + i)
        max_idx = tl.maximum(max_idx, candidate_idx)
    
    # Store the result for this block
    tl.store(block_results_ptr + pid, max_idx)

@triton.jit
def s331_kernel_simple(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    max_idx = -1
    
    # Process all elements
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    
    block_start = 0
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    neg_mask = vals < 0.0
    valid_mask = mask & neg_mask
    
    # Use where to get valid negative indices
    candidate_indices = tl.where(valid_mask, offsets, -1)
    max_idx = tl.max(candidate_indices, axis=0)
    
    # Handle remaining blocks if any
    remaining_start = BLOCK_SIZE
    remaining_offsets = remaining_start + offsets
    remaining_mask = remaining_offsets < n_elements
    
    if remaining_mask[0]:
        remaining_vals = tl.load(a_ptr + remaining_offsets, mask=remaining_mask, other=0.0)
        remaining_neg_mask = remaining_vals < 0.0
        remaining_valid_mask = remaining_mask & remaining_neg_mask
        remaining_candidates = tl.where(remaining_valid_mask, remaining_offsets, -1)
        remaining_max = tl.max(remaining_candidates, axis=0)
        max_idx = tl.maximum(max_idx, remaining_max)
    
    tl.store(result_ptr, max_idx)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Simple approach: process all data with one kernel call
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    # For small arrays, use a simple sequential approach in one thread
    if n_elements <= BLOCK_SIZE:
        grid = (1,)
        s331_kernel_simple[grid](a, result, n_elements, BLOCK_SIZE)
    else:
        # For larger arrays, process in chunks and find global maximum
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        block_results = torch.full((num_blocks,), -1, dtype=torch.int32, device=a.device)
        
        grid = (num_blocks,)
        s331_kernel[grid](a, block_results, n_elements, BLOCK_SIZE)
        
        # Find maximum across all blocks on CPU
        valid_results = block_results[block_results >= 0]
        if len(valid_results) > 0:
            j = torch.max(valid_results).item()
        else:
            j = -1
        result = torch.tensor([j], dtype=torch.int32, device=a.device)
    
    return result[0].item()