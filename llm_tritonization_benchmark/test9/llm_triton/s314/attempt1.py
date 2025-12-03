import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction to find maximum - use single block
    block_id = tl.program_id(0)
    
    if block_id > 0:
        return
    
    # Initialize with first element
    max_val = tl.load(a_ptr)
    
    # Process all elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=-float('inf'))
        
        # Find max within this block
        block_max = tl.max(vals)
        
        # Update global max
        max_val = tl.maximum(max_val, block_max)
    
    # Store result
    tl.store(result_ptr, max_val)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block for reduction
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s314_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()