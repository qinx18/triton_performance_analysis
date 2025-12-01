import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize with first element
    x = tl.load(a_ptr)
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process array in blocks to find maximum
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find maximum in current block
        block_max = tl.max(vals, axis=0)
        
        # Update global maximum
        x = tl.maximum(x, block_max)
    
    # Store result
    tl.store(result_ptr, x)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Output tensor for the maximum value
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with single program
    grid = (1,)
    s314_kernel[grid](
        a, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()