import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    sum_val = 0.0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Sum the block
        block_sum = tl.sum(vals, axis=0)
        sum_val += block_sum
    
    # Store result (only the first thread writes)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, sum_val)

def vsumr_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor for the sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block
    grid = (1,)
    
    vsumr_kernel[grid](
        a_ptr=a,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()