import torch
import triton
import triton.language as tl

@triton.jit
def s316_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load first element as initial minimum
    x = tl.load(a_ptr)
    
    # Process remaining elements in blocks
    for block_start in range(1, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load block of elements
        block_vals = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(block_vals)
        
        # Update global minimum
        x = tl.minimum(x, block_min)
    
    # Store result
    tl.store(output_ptr, x)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread (reduction operation)
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s316_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()