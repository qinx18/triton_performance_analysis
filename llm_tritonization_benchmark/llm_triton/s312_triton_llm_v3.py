import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, prod_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    if block_id == 0:
        prod = 1.0
        
        for block_start in range(0, n_elements, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
            
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    prod *= tl.load(a_ptr + block_start + i)
        
        tl.store(prod_ptr, prod)

def s312_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    prod = torch.ones(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s312_kernel[grid](
        a, prod, n_elements, BLOCK_SIZE
    )
    
    return prod.item()