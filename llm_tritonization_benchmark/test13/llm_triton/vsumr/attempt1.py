import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum for this block
    block_sum = 0.0
    
    # Process all elements using a loop
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from array a
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Add to running sum
        block_sum += tl.sum(vals)
    
    # Store result (only one thread writes the final sum)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, block_sum)

def vsumr_triton(a):
    # Get array size
    n_elements = a.shape[0]
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    vsumr_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()