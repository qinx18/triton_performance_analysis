import torch
import triton
import triton.language as tl

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_prods_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    
    tl.store(partial_prods_ptr + pid, block_prod)

def s312_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    partial_prods = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s312_kernel[grid](a, partial_prods, N, BLOCK_SIZE=BLOCK_SIZE)
    
    while partial_prods.numel() > 1:
        current_size = partial_prods.numel()
        new_num_blocks = triton.cdiv(current_size, BLOCK_SIZE)
        new_partial_prods = torch.ones(new_num_blocks, dtype=a.dtype, device=a.device)
        
        grid = (new_num_blocks,)
        s312_kernel[grid](partial_prods, new_partial_prods, current_size, BLOCK_SIZE=BLOCK_SIZE)
        
        partial_prods = new_partial_prods
    
    return partial_prods.item()