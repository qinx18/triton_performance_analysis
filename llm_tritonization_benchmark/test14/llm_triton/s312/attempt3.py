import torch
import triton
import triton.language as tl

@triton.jit
def _prod_combine(a, b):
    return a * b

@triton.jit
def s312_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_prod = 1.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)
        partial_prod = partial_prod * block_prod
    
    tl.store(output_ptr, partial_prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    output = torch.ones(1, dtype=torch.float32, device=a.device)
    
    grid = (1,)
    s312_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output[0]