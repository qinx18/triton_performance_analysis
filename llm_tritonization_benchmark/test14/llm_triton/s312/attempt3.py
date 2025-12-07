import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
    
    # Compute product using tl.reduce
    block_prod = tl.reduce(vals, axis=0, combine_fn=lambda x, y: x * y)
    
    # Store partial result
    tl.store(result_ptr + pid, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_products = torch.ones(grid_size, dtype=a.dtype, device=a.device)
    
    grid = (grid_size,)
    s312_kernel[grid](
        a, partial_products, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    final_product = torch.prod(partial_products)
    return final_product.item()