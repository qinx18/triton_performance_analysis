import torch
import triton
import triton.language as tl

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_prods_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    block_start = block_idx * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    
    tl.store(partial_prods_ptr + block_idx, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_prods = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s312_kernel[grid](a, partial_prods, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    final_prod = partial_prods[0]
    for i in range(1, num_blocks):
        final_prod = final_prod * partial_prods[i]
    
    return final_prod.item()