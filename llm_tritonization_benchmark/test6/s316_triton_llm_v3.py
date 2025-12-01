import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find minimum value using parallel reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with maximum possible value for this block
    block_min = float('inf')
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values with mask
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_vals_min = tl.min(vals, axis=0)
        block_min = tl.minimum(block_min, block_vals_min)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, block_min)

def s316_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Create output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need global minimum
    grid = (1,)
    
    s316_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()