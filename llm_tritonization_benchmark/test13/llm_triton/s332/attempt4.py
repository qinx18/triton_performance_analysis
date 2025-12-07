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
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in range(num_blocks):
        # Skip if already found (no continue statement)
        if not found:
            block_start = block_id * BLOCK_SIZE
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            # Load block of data
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            
            # Check condition for each element in block
            condition = a_vals > t
            
            # Find first occurrence in this block using tl.where
            valid_condition = condition & mask
            
            # Check each position sequentially
            for i in range(BLOCK_SIZE):
                element_idx = block_start + i
                if element_idx < n_elements:
                    if not found:
                        element_val = tl.load(a_ptr + element_idx)
                        if element_val > t:
                            found_index = element_idx
                            found_value = element_val
                            found = True
    
    # Store results
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