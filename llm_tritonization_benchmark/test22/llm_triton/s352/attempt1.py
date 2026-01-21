import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise products
    products = a_vals * b_vals
    
    # Store products for reduction
    tl.store(output_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Create temporary array for partial products
    temp_products = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Compute element-wise products
    s352_kernel[grid](
        a, b, temp_products,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get dot product
    dot = torch.sum(temp_products)
    
    return dot