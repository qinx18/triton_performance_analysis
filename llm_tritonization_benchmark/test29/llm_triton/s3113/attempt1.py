import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block maximum with negative infinity
    block_max = float('-inf')
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute absolute values and find max in this block
        abs_vals = tl.abs(vals)
        
        # Update block maximum
        current_max = tl.max(abs_vals)
        block_max = tl.maximum(block_max, current_max)
    
    # Store the result
    tl.store(output_ptr, block_max)

def s3113_triton(a, abs):
    n_elements = a.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block to handle reduction
    grid = (1,)
    s3113_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()