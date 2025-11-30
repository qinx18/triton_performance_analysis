import triton
import triton.language as tl
import torch

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the first element > t in array a
    # Each program handles one block of elements
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Check which elements are > t
    condition = a_vals > t
    
    # Find first occurrence in this block
    found_mask = condition
    
    # Use cumsum to find the first occurrence
    # Convert boolean to int for cumsum
    found_int = found_mask.to(tl.int32)
    cumsum = tl.cumsum(found_int, axis=0)
    
    # First occurrence has cumsum == 1
    first_occurrence = (cumsum == 1) & found_mask
    
    # Get the local index within block where first occurrence happens
    local_indices = tl.arange(0, BLOCK_SIZE)
    
    # Use where to get the first valid index, -1 if none found
    found_any = tl.sum(first_occurrence.to(tl.int32)) > 0
    
    if found_any:
        # Find the position of the first occurrence
        first_pos = tl.sum(tl.where(first_occurrence, local_indices, BLOCK_SIZE))
        global_index = block_start + first_pos
        
        # Get the value at that position
        if global_index < n_elements:
            found_value = tl.load(a_ptr + global_index)
            # Store result as [index, value, found_flag]
            tl.store(result_ptr + pid * 3, global_index.to(tl.float32))
            tl.store(result_ptr + pid * 3 + 1, found_value)
            tl.store(result_ptr + pid * 3 + 2, 1.0)  # found flag
        else:
            tl.store(result_ptr + pid * 3, -2.0)
            tl.store(result_ptr + pid * 3 + 1, -1.0)
            tl.store(result_ptr + pid * 3 + 2, 0.0)  # not found flag
    else:
        # No element found in this block
        tl.store(result_ptr + pid * 3, -2.0)
        tl.store(result_ptr + pid * 3 + 1, -1.0)
        tl.store(result_ptr + pid * 3 + 2, 0.0)  # not found flag

def s332_triton(a, t_val):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create result tensor to store [index, value, found_flag] for each block
    result = torch.zeros((grid_size, 3), dtype=torch.float32, device=a.device)
    
    s332_kernel[(grid_size,)](
        a, t_val, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find the first block that found a result
    found_flags = result[:, 2]
    found_blocks = torch.nonzero(found_flags, as_tuple=True)[0]
    
    if len(found_blocks) > 0:
        # Get the first block that found something
        first_block = found_blocks[0]
        
        # Among all blocks that found something, find the one with smallest index
        valid_results = result[found_flags > 0]
        min_idx_pos = torch.argmin(valid_results[:, 0])
        
        final_index = int(valid_results[min_idx_pos, 0].item())
        final_value = valid_results[min_idx_pos, 1].item()
    else:
        final_index = -2
        final_value = -1.0
    
    chksum = final_value + float(final_index)
    
    return final_value