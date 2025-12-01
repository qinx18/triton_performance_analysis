import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find minimum value using reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize minimum with first element
    if tl.program_id(0) == 0:
        min_val = tl.load(a_ptr)
    else:
        min_val = float('inf')
    
    # Process array in blocks
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Update minimum
        min_val = tl.minimum(min_val, tl.min(a_vals))
    
    # Store result from first program
    if tl.program_id(0) == 0:
        tl.store(result_ptr, min_val)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s316_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()