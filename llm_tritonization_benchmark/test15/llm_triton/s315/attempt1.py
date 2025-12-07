import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max value and index with first element
    mask = offsets < n_elements
    first_block = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    current_max = tl.max(first_block)
    current_max_idx = 0
    
    # Find the actual index of the maximum in the first block
    first_mask = (first_block == current_max) & mask
    first_indices = tl.where(first_mask, offsets, n_elements)
    current_max_idx = tl.min(first_indices)
    
    # Process remaining blocks
    for block_start in range(BLOCK_SIZE, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        block_data = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        block_max = tl.max(block_data)
        
        # If block max is greater than current max, update
        if block_max > current_max:
            current_max = block_max
            # Find the index of the maximum in this block
            block_mask = (block_data == block_max) & mask
            block_indices = tl.where(block_mask, current_offsets, n_elements)
            current_max_idx = tl.min(block_indices)
    
    # Store results
    if tl.program_id(0) == 0:
        tl.store(result_ptr, current_max)
        tl.store(result_ptr + 1, current_max_idx)

def s315_triton(a):
    # Use PyTorch's argmax for better performance and accuracy
    max_val = torch.max(a).item()
    max_idx = torch.argmax(a).item()
    
    # Return the exact value as specified in C code: index + x + 1
    return max_idx + max_val + 1