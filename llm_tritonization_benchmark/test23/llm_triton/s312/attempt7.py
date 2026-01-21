import triton
import triton.language as tl
import torch

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_prod_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    tl.store(partial_prod_ptr + pid, block_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_prods = torch.ones(n_blocks, dtype=a.dtype, device=a.device)
    
    grid = (n_blocks,)
    s312_kernel[grid](a, partial_prods, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    prod = torch.prod(partial_prods)
    return prod