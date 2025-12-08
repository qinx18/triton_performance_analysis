import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with negative infinity for max reduction
    block_max = float('-inf')
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values with masking
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Update block maximum
        current_max = tl.max(vals)
        block_max = tl.maximum(block_max, current_max)
    
    # Store the result (each work item stores the same value)
    tl.store(result_ptr, block_max)

def s314_triton(a):
    n_elements = a.numel()
    
    # Create output tensor
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with single work item since this is a global reduction
    BLOCK_SIZE = 256
    grid = (1,)
    
    s314_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()