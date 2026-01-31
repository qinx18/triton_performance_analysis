import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block maximum to negative infinity
    block_max = float('-inf')
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Update block maximum
        current_max = tl.max(vals, axis=0)
        block_max = tl.maximum(block_max, current_max)
    
    # Store result
    tl.store(result_ptr, block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Allocate output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program
    grid = (1,)
    s314_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()