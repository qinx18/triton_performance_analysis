import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element value
    first_val = tl.load(a_ptr)
    block_min = first_val
    
    # Process blocks starting from index 1
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.minimum(block_min, tl.min(vals, axis=0))
    
    # Store the minimum value found
    tl.store(result_ptr, block_min)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with appropriate block size
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s316_kernel[grid](
        a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()