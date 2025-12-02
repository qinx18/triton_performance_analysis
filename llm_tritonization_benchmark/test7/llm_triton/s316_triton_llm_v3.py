import torch
import triton
import triton.language as tl

@triton.jit
def s316_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find minimum value across all elements using parallel reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize local minimum to positive infinity
    local_min = float('inf')
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(vals, axis=0)
        
        # Update local minimum
        local_min = tl.minimum(local_min, block_min)
    
    # Store the result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, local_min)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor for the minimum value
    output = torch.empty(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need global minimum
    grid = (1,)
    
    s316_kernel[grid](
        a, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output[0].item()