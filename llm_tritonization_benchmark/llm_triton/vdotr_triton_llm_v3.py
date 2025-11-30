import triton
import triton.language as tl
import torch

@triton.jit
def vdotr_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    products = a_vals * b_vals
    
    tl.store(output_ptr + offsets, products, mask=mask)

def vdotr_triton(a, b):
    n_elements = a.shape[0]
    
    # Create temporary array to store products
    products = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vdotr_kernel[grid](
        a, b, products,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum all products to get final dot product
    dot = torch.sum(products)
    
    return dot.item()