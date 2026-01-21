import triton
import triton.language as tl
import torch

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_results_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values, use 1.0 as identity for product reduction
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
    
    # Reduce within the block
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    
    # Store partial result
    tl.store(partial_results_ptr + block_id, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create tensor for partial results
    partial_results = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s312_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Combine partial results using torch.prod
    final_result = torch.prod(partial_results)
    
    return final_result.item()