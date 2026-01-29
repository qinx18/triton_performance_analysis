import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block maximum to negative infinity
    block_max = float('-inf')
    
    # Process all blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values with negative infinity for out-of-bounds
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Update block maximum
        current_max = tl.max(vals)
        block_max = tl.maximum(block_max, current_max)
    
    # Store the final maximum
    tl.store(result_ptr, block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    BLOCK_SIZE = 1024
    s314_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()