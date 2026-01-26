import triton
import triton.language as tl
import torch

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program id
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask, other=float('-inf'))
    
    # Find elements greater than threshold
    greater_mask = a_vals > t
    
    # If any element is greater than t, find the first one
    if tl.sum(greater_mask.to(tl.int32)) > 0:
        # Create position array within this block
        positions = indices
        
        # Set positions to large value where condition is not met
        filtered_positions = tl.where(greater_mask, positions, n_elements)
        
        # Find minimum position (first occurrence)
        min_pos = tl.min(filtered_positions)
        
        # Check if this minimum position is valid and in our block
        if min_pos < n_elements:
            # Get the value at minimum position
            first_match_mask = positions == min_pos
            value = tl.sum(tl.where(first_match_mask, a_vals, 0.0))
            
            # Store result: [found_flag, index, value]
            tl.store(result_ptr + pid * 3 + 0, 1.0)  # found flag
            tl.store(result_ptr + pid * 3 + 1, min_pos)  # index
            tl.store(result_ptr + pid * 3 + 2, value)  # value
        else:
            # No match found in this block
            tl.store(result_ptr + pid * 3 + 0, 0.0)
            tl.store(result_ptr + pid * 3 + 1, -2.0)
            tl.store(result_ptr + pid * 3 + 2, -1.0)
    else:
        # No match found in this block
        tl.store(result_ptr + pid * 3 + 0, 0.0)
        tl.store(result_ptr + pid * 3 + 1, -2.0)
        tl.store(result_ptr + pid * 3 + 2, -1.0)

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create result tensor to store [found_flag, index, value] for each block
    result = torch.zeros((num_blocks * 3,), dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s332_kernel[(num_blocks,)](
        a, t, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Process results to find the first occurrence across all blocks
    results_reshaped = result.view(num_blocks, 3)
    found_flags = results_reshaped[:, 0]
    indices = results_reshaped[:, 1]
    values = results_reshaped[:, 2]
    
    # Find blocks that found matches
    found_blocks = torch.nonzero(found_flags).flatten()
    
    if len(found_blocks) > 0:
        # Among blocks that found matches, find the one with minimum index
        found_indices = indices[found_blocks]
        found_values = values[found_blocks]
        
        min_idx_pos = torch.argmin(found_indices)
        final_value = found_values[min_idx_pos]
        
        return final_value
    else:
        # No match found anywhere
        return torch.tensor(-1.0, dtype=torch.float32, device=a.device)