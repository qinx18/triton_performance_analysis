import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Only the first program should execute
    if pid != 0:
        return
        
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result values
    found_index = -2
    found_value = -1.0
    found = False
    
    # Process blocks sequentially to maintain order
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from array a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check condition a[i] > t
        condition = a_vals > t
        
        # Find first occurrence in this block
        for i in range(BLOCK_SIZE):
            local_offset = block_start + i
            if local_offset >= n_elements:
                found = True
            elif not found:
                a_val = tl.load(a_ptr + local_offset)
                if a_val > t:
                    found_index = local_offset
                    found_value = a_val
                    found = True
    
    # Store results
    chksum = found_value + found_index
    tl.store(result_ptr, found_value)
    tl.store(result_ptr + 1, found_index)
    tl.store(result_ptr + 2, chksum)

def s332_triton(a, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor to store [value, index, chksum]
    result = torch.zeros(3, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block to maintain sequential order
    grid = (1,)
    s332_kernel[grid](
        a, result, t, n_elements, BLOCK_SIZE
    )
    
    return result[0].item()  # Return the found value