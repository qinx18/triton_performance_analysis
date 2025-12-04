import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    values = tl.load(a_ptr + current_offsets, mask=mask, other=t)
    
    # Check condition
    condition = values > t
    condition = condition & mask
    
    # Find first match in this block
    found_any = tl.sum(condition.to(tl.int32)) > 0
    
    # Initialize results
    found_index = -2
    found_value = -1.0
    
    # Use vectorized approach to find first match
    if found_any:
        # Create position indices for current block
        positions = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Use large value for non-matches to find minimum valid position
        valid_positions = tl.where(condition, positions, n_elements + 1000000)
        min_pos = tl.min(valid_positions)
        
        # If we found a valid position, extract the corresponding value
        if min_pos < n_elements:
            found_index = min_pos
            # Find which element in our block corresponds to min_pos
            match_mask = (positions == min_pos) & mask
            found_value = tl.sum(tl.where(match_mask, values, 0.0))
    
    # Store results
    result_offset = block_id * 4
    tl.store(result_ptr + result_offset, found_index.to(tl.float32))
    tl.store(result_ptr + result_offset + 1, found_value)
    tl.store(result_ptr + result_offset + 2, found_any.to(tl.float32))
    tl.store(result_ptr + result_offset + 3, block_id.to(tl.float32))

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Result tensor: [index, value, found_flag, block_id] for each block
    result = torch.zeros((num_blocks * 4,), device=a.device, dtype=torch.float32)
    
    # Launch kernel
    grid = (num_blocks,)
    s332_kernel[grid](
        a, result, t, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Process results to find first match across all blocks
    result_reshaped = result.view(num_blocks, 4)
    found_flags = result_reshaped[:, 2]
    
    # Find blocks that found matches
    found_blocks = torch.nonzero(found_flags, as_tuple=False).flatten()
    
    if len(found_blocks) == 0:
        # No match found
        value = -1.0
    else:
        # Get the first block that found a match
        first_block = found_blocks[0].item()
        value = result_reshaped[first_block, 1].item()
    
    return value