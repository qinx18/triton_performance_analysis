import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    products = a_vals * b_vals
    tl.store(result_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    n = a.shape[0]
    device = a.device
    
    # Create temporary array to store element-wise products
    products = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s352_kernel[grid](
        a, b, products, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final dot product
    dot = torch.sum(products)
    return dot.item()