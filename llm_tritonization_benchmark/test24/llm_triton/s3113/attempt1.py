import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize running maximum
    running_max = 0.0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute absolute values and find block maximum
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals)
        
        # Update running maximum
        running_max = tl.maximum(running_max, block_max)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, running_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor for the maximum value
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block to handle reduction
    
    s3113_kernel[grid](
        a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()