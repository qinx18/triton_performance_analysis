import triton
import triton.language as tl
import torch

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_prods, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_prod = 1.0
    
    for start in range(block_start, n, BLOCK_SIZE * tl.num_programs(0)):
        current_offsets = start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        vals_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
        block_prod = block_prod * vals_prod
    
    tl.store(partial_prods + pid, block_prod)

def s312_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = min(triton.cdiv(n, BLOCK_SIZE), 1024)
    
    partial_prods = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s312_kernel[grid](a, partial_prods, n, BLOCK_SIZE=BLOCK_SIZE)
    
    result = torch.prod(partial_prods).item()
    return result