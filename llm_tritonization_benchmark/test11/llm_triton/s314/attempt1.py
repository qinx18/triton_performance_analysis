import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find maximum value in array a using reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize local maximum with first element (or -inf for empty blocks)
    local_max = float('-inf')
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Update local maximum
        block_max = tl.max(vals)
        local_max = tl.maximum(local_max, block_max)
    
    # Store result (all threads store the same value)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, local_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch single block kernel since we need global reduction
    grid = (1,)
    
    s314_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()