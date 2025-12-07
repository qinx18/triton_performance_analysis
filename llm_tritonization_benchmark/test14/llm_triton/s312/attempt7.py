import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, partial_results_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
    
    # Compute product within this block
    block_prod = 1.0
    for i in range(BLOCK_SIZE):
        if i < BLOCK_SIZE:
            block_prod = block_prod * vals[i]
    
    # Store partial result
    tl.store(partial_results_ptr + pid, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_results = torch.ones(grid_size, dtype=a.dtype, device=a.device)
    
    grid = (grid_size,)
    s312_kernel[grid](
        a, partial_results, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Combine partial results on CPU
    final_product = torch.prod(partial_results)
    return final_product.item()