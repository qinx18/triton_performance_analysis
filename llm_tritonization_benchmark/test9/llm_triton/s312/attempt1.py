import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel - each program computes partial product
    pid = tl.program_id(0)
    
    # Calculate range for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=1.0)
    
    # Compute product within block
    block_prod = tl.reduce(a_vals, axis=0, combine_fn=lambda x, y: x * y)
    
    # Store partial result
    if tl.program_id(0) == 0:
        tl.store(result_ptr + pid, block_prod)
    else:
        tl.store(result_ptr + pid, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary buffer for partial results
    partial_results = torch.ones(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel for partial products
    grid = (n_blocks,)
    s312_kernel[grid](
        a, partial_results, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reduce partial results on CPU for final product
    final_prod = torch.prod(partial_results)
    
    return final_prod.item()