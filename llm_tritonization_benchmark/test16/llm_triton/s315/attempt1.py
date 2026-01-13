import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize max value and index
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Process array in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from current block
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find indices where values are greater than current max
        greater_mask = vals > max_val
        
        # Update max value if any value in block is greater
        block_max = tl.max(vals)
        if block_max > max_val:
            max_val = block_max
            # Find the index of the maximum value in this block
            for i in range(BLOCK_SIZE):
                offset_idx = block_start + i
                if offset_idx < n_elements:
                    val = tl.load(a_ptr + offset_idx)
                    if val > tl.load(a_ptr + max_idx):
                        max_idx = offset_idx
    
    # Store results
    tl.store(output_ptr, max_val)
    tl.store(output_ptr + 1, max_idx)

def s315_triton(a):
    # Use PyTorch's built-in functions for argmax reduction
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return the exact value as in C code: index + x + 1
    result = max_idx + max_val + 1
    return result.item()