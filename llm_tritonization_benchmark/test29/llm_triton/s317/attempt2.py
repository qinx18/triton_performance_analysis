import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program computes the product for n/2 iterations
    # Since this is a reduction, we'll have each block compute the full result
    if pid > 0:
        return
    
    # Initialize q
    q = 1.0
    
    # Compute the number of iterations
    num_iterations = n // 2
    
    # Perform the product reduction
    factor = 0.99
    
    # Process iterations in blocks to avoid long loops
    iterations_per_block = BLOCK_SIZE
    full_blocks = num_iterations // iterations_per_block
    remaining = num_iterations % iterations_per_block
    
    # Process full blocks
    for block_idx in range(full_blocks):
        # Apply factor^BLOCK_SIZE
        block_factor = factor
        for _ in range(iterations_per_block):
            block_factor = block_factor * factor
        q = q * block_factor
    
    # Process remaining iterations
    remaining_factor = 1.0
    for _ in range(remaining):
        remaining_factor = remaining_factor * factor
    q = q * remaining_factor
    
    # Store result
    tl.store(output_ptr, q)

def s317_triton(n):
    # Create output tensor
    output = torch.zeros(1, dtype=torch.float32, device='cuda')
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Launch kernel with single block since this is a scalar reduction
    grid = (1,)
    
    s317_kernel[grid](
        output,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()