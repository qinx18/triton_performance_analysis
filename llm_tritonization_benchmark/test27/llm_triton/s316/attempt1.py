import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block minimum to positive infinity
    block_min = float('inf')
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values, using inf as other value for masked elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Update block minimum
        current_min = tl.min(vals)
        if current_min < block_min:
            block_min = current_min
    
    # Store result
    tl.store(result_ptr, block_min)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s316_kernel[grid](
        a, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()