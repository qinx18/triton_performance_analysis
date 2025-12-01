import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Initialize block maximum with very small value
    block_max = -1e30
    
    # Process elements in chunks
    for chunk_start in range(0, n_elements, BLOCK_SIZE):
        chunk_offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = chunk_offsets < n_elements
        
        # Load values with masking
        values = tl.load(a_ptr + chunk_offsets, mask=mask, other=-1e30)
        
        # Compute absolute values
        abs_values = tl.abs(values)
        
        # Update block maximum
        block_max = tl.maximum(block_max, tl.max(abs_values))
    
    # Store the block maximum
    if pid == 0:
        tl.store(output_ptr, block_max)
    else:
        # Use atomic max to find global maximum
        tl.atomic_max(output_ptr, block_max)

def s3113_triton(a):
    n_elements = a.numel()
    
    # Create output tensor
    output = torch.tensor([-1e30], device=a.device, dtype=a.dtype)
    
    # Launch kernel with single block to handle reduction
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s3113_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()