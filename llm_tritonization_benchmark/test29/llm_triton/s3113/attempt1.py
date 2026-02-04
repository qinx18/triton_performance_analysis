import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate number of blocks needed
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # Initialize global maximum with absolute value of first element
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        tl.store(output_ptr, tl.abs(first_val))
    
    # Process all elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values and compute absolute values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Find block maximum
        block_max = tl.max(abs_vals, axis=0)
        
        # Update global maximum atomically
        current_max = tl.load(output_ptr)
        if block_max > current_max:
            tl.atomic_max(output_ptr, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor for the maximum value
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s3113_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()