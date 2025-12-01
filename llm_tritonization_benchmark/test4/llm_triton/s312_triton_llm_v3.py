import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize product to 1.0
    prod = 1.0
    
    # Process array in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load block of data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Multiply all elements in the block
        block_prod = tl.reduce(a_vals, axis=0, op=tl.math.multiply_op)
        
        # Update total product
        prod *= block_prod
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, prod)

def s312_triton(a):
    n = a.shape[0]
    result = torch.ones(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s312_kernel[grid](
        a, result, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()