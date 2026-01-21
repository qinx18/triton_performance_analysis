import torch
import triton
import triton.language as tl

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_prods, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    vals = tl.load(a_ptr + indices, mask=mask, other=1.0)
    
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    tl.store(partial_prods + pid, block_prod)

def s312_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    partial_prods = torch.ones(n_blocks, dtype=a.dtype, device=a.device)
    
    grid = (n_blocks,)
    s312_kernel[grid](a, partial_prods, N, BLOCK_SIZE=BLOCK_SIZE)
    
    prod = torch.prod(partial_prods)
    return prod