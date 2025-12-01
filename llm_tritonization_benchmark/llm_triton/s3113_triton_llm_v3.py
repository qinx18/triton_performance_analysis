import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global max using block-wise reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block max to negative infinity
    block_max = float('-inf')
    
    # Process all blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values with mask
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Compute absolute values
        abs_vals = tl.abs(vals)
        
        # Find max in this block
        current_block_max = tl.max(abs_vals, axis=0)
        
        # Update global max
        block_max = tl.maximum(block_max, current_block_max)
    
    # Store result (each thread will store the same value, but that's ok for a single output)
    tl.store(output_ptr, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with single block that processes all elements
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    grid = (1,)
    s3113_kernel[grid](
        a_ptr=a,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.item()