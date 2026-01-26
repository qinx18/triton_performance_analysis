import torch
import triton
import triton.language as tl

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_prods_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
    
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    tl.store(partial_prods_ptr + block_id, block_prod)

def s312_triton(a):
    N = a.shape[0]
    
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    partial_prods = torch.empty(n_blocks, dtype=a.dtype, device=a.device)
    
    s312_kernel[(n_blocks,)](
        a, 
        partial_prods, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    prod = torch.prod(partial_prods)
    return prod