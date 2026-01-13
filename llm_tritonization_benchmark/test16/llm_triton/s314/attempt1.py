import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    block_max = first_val
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.maximum(block_max, tl.max(vals, axis=0))
    
    # Store result
    if tl.program_id(0) == 0:
        tl.store(result_ptr, block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s314_kernel[grid](
        a, result,
        n_elements,
        BLOCK_SIZE
    )
    
    return result.item()