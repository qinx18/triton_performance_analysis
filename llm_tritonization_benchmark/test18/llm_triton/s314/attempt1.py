import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find maximum value in array a
    max_val = float('-inf')
    
    # Process array in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find block maximum
        block_max = tl.max(vals, axis=0)
        
        # Update global maximum
        max_val = tl.maximum(max_val, block_max)
    
    # Store result
    tl.store(result_ptr, max_val)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program
    s314_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()