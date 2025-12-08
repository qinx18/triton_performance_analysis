import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max value and index
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Process blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in this block
        block_max = tl.max(vals)
        
        # Update global max if needed
        if block_max > max_val:
            max_val = block_max
            # Find the index of max in this block
            max_mask = vals == block_max
            # Get first occurrence index
            indices = tl.where(max_mask, current_offsets, n_elements)
            max_idx = tl.min(indices)
    
    # Store result
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx)

def s315_triton(a):
    # Use PyTorch for argmax reduction as recommended
    max_val = torch.max(a).item()
    max_idx = torch.argmax(a).item()
    
    # Return the exact value as in C code: index + x + 1
    return max_idx + max_val + 1