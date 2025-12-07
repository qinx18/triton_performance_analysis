import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_index_ptr, result_value_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result values
    found_index = -2
    found_value = -1.0
    found = False
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check condition
        condition = a_vals > t
        
        # Find first occurrence in this block
        for i in range(BLOCK_SIZE):
            if not found:
                offset_i = block_start + i
                if offset_i < n_elements:
                    val_i = tl.load(a_ptr + offset_i)
                    if val_i > t:
                        found_index = offset_i
                        found_value = val_i
                        found = True
        
        # If found in this block, break
        if found:
            break
    
    # Store results (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_index_ptr, found_index)
        tl.store(result_value_ptr, found_value)

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Create result tensors
    result_index = torch.tensor(-2, dtype=torch.int32, device=a.device)
    result_value = torch.tensor(-1.0, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s332_kernel[grid](
        a, t, result_index, result_value, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Calculate chksum
    chksum = result_value + result_index.float()
    
    return result_value, result_index, chksum