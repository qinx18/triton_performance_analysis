import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single block handles the entire reduction
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max with absolute value of first element
    first_val = tl.load(a_ptr, mask=True)
    running_max = tl.abs(first_val)
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals, axis=0)
        
        # Update running maximum
        running_max = tl.maximum(running_max, block_max)
    
    # Store final result
    tl.store(output_ptr, running_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor to store the maximum value
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Launch single block to handle the reduction
    grid = (1,)
    
    s3113_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()