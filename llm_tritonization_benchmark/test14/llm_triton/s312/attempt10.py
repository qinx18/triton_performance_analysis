import torch
import triton
import triton.language as tl

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, partial_results_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_id * BLOCK_SIZE + offsets
    
    mask = current_offsets < n_elements
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
    
    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
    
    tl.store(partial_results_ptr + block_id, block_prod)

def s312_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    if n_elements == 0:
        return 1.0
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_results = torch.ones(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s312_kernel[grid](
        a, partial_results, n_elements, BLOCK_SIZE
    )
    
    result = torch.prod(partial_results)
    return result.item()