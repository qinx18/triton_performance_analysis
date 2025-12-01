import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize result with first element
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        tl.store(result_ptr, first_val)
    
    # Define offsets once before the loop
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks of the array
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block
        block_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in current block
        block_max = tl.max(block_vals)
        
        # Atomic max with global result
        if block_max != float('-inf'):
            tl.atomic_max(result_ptr, block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create result tensor initialized with negative infinity
    result = torch.tensor([float('-inf')], device=a.device, dtype=a.dtype)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s314_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()