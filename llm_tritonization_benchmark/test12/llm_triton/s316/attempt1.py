import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find minimum value using parallel reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize minimum with first valid element
    min_val = float('inf')
    
    # Process all blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(vals)
        
        # Update global minimum
        min_val = tl.minimum(min_val, block_min)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, min_val)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need global reduction
    grid = (1,)
    s316_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()