import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with positive infinity for blocks that might be partially empty
    current_min = float('inf')
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values, using inf for masked elements so they don't affect min
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(vals, axis=0)
        
        # Update global minimum
        current_min = tl.minimum(current_min, block_min)
    
    # Store the final minimum
    tl.store(result_ptr, current_min)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program instance
    s316_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()