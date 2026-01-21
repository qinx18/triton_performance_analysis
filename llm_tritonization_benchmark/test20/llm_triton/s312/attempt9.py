import triton
import triton.language as tl
import torch

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_products, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n
    
    vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=1.0)
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    
    tl.store(partial_products + pid, block_prod)

def s312_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    partial_products = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s312_kernel[grid](a, partial_products, n, BLOCK_SIZE=BLOCK_SIZE)
    
    prod = torch.prod(partial_products)
    return prod