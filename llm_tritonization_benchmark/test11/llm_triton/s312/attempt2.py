import triton
import triton.language as tl
import torch

@triton.jit
def s312_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    prod = 1.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        # Use reduction instead of scalar indexing
        block_prod = tl.reduce(vals, axis=0, combine_fn=lambda x, y: x * y)
        prod = prod * block_prod
    
    tl.store(result_ptr, prod)

def s312_triton(a):
    BLOCK_SIZE = 1024
    
    result = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s312_kernel[grid](
        a, result,
        a.numel(),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()