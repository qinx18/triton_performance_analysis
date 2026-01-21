import triton
import triton.language as tl
import torch

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_prods_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    vals = tl.load(a_ptr + indices, mask=mask, other=1.0)
    
    # Only include values that are actually within bounds for the product
    masked_vals = tl.where(mask, vals, 1.0)
    block_prod = tl.reduce(masked_vals, axis=0, combine_fn=_prod_combine)
    
    tl.store(partial_prods_ptr + block_id, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_prods = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s312_kernel[grid](a, partial_prods, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    final_prod = torch.prod(partial_prods)
    
    return final_prod.item()