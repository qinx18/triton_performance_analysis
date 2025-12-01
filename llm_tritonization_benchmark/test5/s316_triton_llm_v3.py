import torch
import triton
import triton.language as tl

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize with first element
    min_val = tl.load(a_ptr)
    
    # Define offsets once at start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process remaining elements in blocks
    for block_start in range(BLOCK_SIZE, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(vals, axis=0)
        
        # Update global minimum
        min_val = tl.minimum(min_val, block_min)
    
    # Handle remaining elements if n_elements-1 is not divisible by BLOCK_SIZE
    remaining_start = ((n_elements - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if remaining_start < n_elements - 1:
        remaining_offsets = remaining_start + offsets
        mask = (remaining_offsets < n_elements) & (remaining_offsets > 0)
        if tl.sum(mask.to(tl.int32)) > 0:
            vals = tl.load(a_ptr + remaining_offsets, mask=mask, other=float('inf'))
            block_min = tl.min(vals, axis=0)
            min_val = tl.minimum(min_val, block_min)
    
    # Store result
    tl.store(result_ptr, min_val)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread block since we need global reduction
    grid = (1,)
    s316_kernel[grid](
        a, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()