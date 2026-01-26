import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    dot_accum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        products = a_vals * b_vals
        dot_accum += tl.sum(products)
    
    tl.store(dot_ptr, dot_accum)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    dot_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s352_kernel[grid](
        a, b, dot_result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dot_result.item()