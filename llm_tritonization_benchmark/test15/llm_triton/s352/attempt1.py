import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    products = a_vals * b_vals
    
    tl.store(result_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for products
    products = torch.zeros_like(a)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s352_kernel[grid](
        a, b, products, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final dot product
    dot = torch.sum(products)
    
    return dot.item()