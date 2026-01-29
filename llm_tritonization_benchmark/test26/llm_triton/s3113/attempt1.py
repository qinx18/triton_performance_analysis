import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block max with first element's absolute value
    first_val = tl.load(a_ptr, mask=True)
    block_max = tl.abs(first_val)
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute absolute values and find max in this block
        abs_vals = tl.abs(vals)
        current_block_max = tl.max(abs_vals, axis=0)
        
        # Update global maximum
        block_max = tl.maximum(block_max, current_block_max)
    
    # Store result
    tl.store(result_ptr, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s3113_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()