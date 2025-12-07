import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum for this block
    block_sum = 0.0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from array a
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Sum the values in this block
        block_sum += tl.sum(vals, axis=0)
    
    # Store the result (only thread 0 writes the final sum)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, block_sum)

def vsumr_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor to store the sum
    output = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with a single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    vsumr_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()