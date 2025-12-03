import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    sum_val = 0.0
    
    # Loop over blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Add to sum
        sum_val += tl.sum(vals)
    
    # Store result (only first thread stores)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, sum_val)

def s311_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    BLOCK_SIZE = 256
    grid = (1,)
    
    s311_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result[0].item()