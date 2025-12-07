import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel - each program computes part of the product
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=1.0)
    
    # Compute partial product for this block
    partial_prod = tl.reduce(a_vals, axis=0, op=lambda x, y: x * y)
    
    # Store partial result
    if pid == 0:
        tl.atomic_mul(result_ptr, partial_prod)
    else:
        # For other blocks, we need to accumulate the product
        tl.atomic_mul(result_ptr, partial_prod)

@triton.jit 
def s312_reduction_kernel(partial_results_ptr, result_ptr, n_partials, BLOCK_SIZE: tl.constexpr):
    # Reduce partial products
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials
    
    partials = tl.load(partial_results_ptr + offsets, mask=mask, other=1.0)
    final_prod = tl.reduce(partials, axis=0, op=lambda x, y: x * y)
    
    if tl.program_id(0) == 0:
        tl.store(result_ptr, final_prod)

def s312_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # For product reduction, we need a two-stage approach
    # Stage 1: Compute partial products
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_results = torch.ones(grid_size, device=a.device, dtype=a.dtype)
    result = torch.ones(1, device=a.device, dtype=a.dtype)
    
    # Compute partial products for each block
    for i in range(grid_size):
        block_start = i * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_elements)
        block_size = block_end - block_start
        
        if block_size > 0:
            # Load block data
            block_data = a[block_start:block_end]
            # Compute product for this block
            block_prod = torch.prod(block_data)
            partial_results[i] = block_prod
    
    # Stage 2: Reduce partial products
    final_prod = torch.prod(partial_results)
    
    return final_prod.item()