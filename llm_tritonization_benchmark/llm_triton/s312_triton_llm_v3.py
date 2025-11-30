import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction operation - compute product of all elements in array a
    # Each block processes BLOCK_SIZE elements and computes partial product
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements with mask, use 1.0 as default for out-of-bounds
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    
    # Compute partial product for this block
    partial_prod = tl.reduce(a_vals, axis=0, combine_fn=lambda x, y: x * y)
    
    # Store partial result
    tl.store(result_ptr + tl.program_id(0), partial_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary array to store partial products
    partial_results = torch.ones(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute partial products
    grid = (n_blocks,)
    s312_kernel[grid](
        a, partial_results, n_elements, BLOCK_SIZE
    )
    
    # Compute final product by multiplying all partial results on CPU
    # For numerical stability, we do this reduction on GPU as well if needed
    while partial_results.shape[0] > 1:
        n_partials = partial_results.shape[0]
        n_blocks_next = triton.cdiv(n_partials, BLOCK_SIZE)
        next_partials = torch.ones(n_blocks_next, dtype=a.dtype, device=a.device)
        
        grid = (n_blocks_next,)
        s312_kernel[grid](
            partial_results, next_partials, n_partials, BLOCK_SIZE
        )
        
        partial_results = next_partials
    
    return partial_results[0]