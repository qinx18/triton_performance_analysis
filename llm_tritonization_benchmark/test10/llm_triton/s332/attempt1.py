import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize values for this block
    found_index = -2
    found_value = -1.0
    found = False
    
    # Process elements in this block
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    values = tl.load(a_ptr + current_offsets, mask=mask, other=t)  # Use t as default to ensure condition fails
    
    # Check condition
    condition = values > t
    condition = condition & mask
    
    # Find first match in this block
    if tl.sum(condition.to(tl.int32)) > 0:
        # There's at least one match
        for i in range(BLOCK_SIZE):
            if current_offsets[i] < n_elements:
                if values[i] > t:
                    found_index = current_offsets[i]
                    found_value = values[i]
                    found = True
                    break
    
    # Store results (index, value, found_flag, block_id for ordering)
    result_offset = block_id * 4
    tl.store(result_ptr + result_offset, found_index.to(tl.float32))
    tl.store(result_ptr + result_offset + 1, found_value)
    tl.store(result_ptr + result_offset + 2, found.to(tl.float32))
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
        index = -2
        value = -1.0
    else:
        # Get the first block that found a match
        first_block = found_blocks[0].item()
        index = int(result_reshaped[first_block, 0].item())
        value = result_reshaped[first_block, 1].item()
    
    chksum = value + float(index)
    
    return value