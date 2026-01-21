import triton
import triton.language as tl
import torch

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for this block
    block_prod = 1.0
    
    # Process elements in chunks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values, use 1.0 as identity for product reduction
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Reduce within the loaded block
        chunk_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
        
        # Accumulate into block product
        block_prod = block_prod * chunk_prod
    
    # Store the result
    tl.store(result_ptr, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor for the result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need to compute total product
    grid = (1,)
    s312_kernel[grid](a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()