import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find minimum value across all elements
    min_val = tl.load(a_ptr)  # Initialize with a[0]
    
    # Process array in blocks to find minimum
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        block_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(block_vals)
        
        # Update global minimum
        min_val = tl.minimum(min_val, block_min)
    
    # Store result
    tl.store(result_ptr, min_val)

def s316_triton(a):
    n_elements = a.numel()
    
    # Create output tensor for result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s316_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()