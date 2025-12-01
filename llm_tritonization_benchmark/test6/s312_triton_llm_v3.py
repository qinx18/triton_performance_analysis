import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize product as 1.0
    prod = 1.0
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Multiply all values in the block
        block_prod = tl.reduce(vals, 0, tl.mul)
        
        # Accumulate with running product
        prod *= block_prod
    
    # Store final result
    if tl.program_id(0) == 0:
        tl.store(result_ptr, prod)

def s312_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Launch kernel with single program
    BLOCK_SIZE = 256
    grid = (1,)
    
    s312_kernel[grid](
        a, result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()