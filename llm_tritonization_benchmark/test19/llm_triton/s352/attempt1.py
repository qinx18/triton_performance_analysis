import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load values from arrays a and b
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise product
    products = a_vals * b_vals
    
    # Store the products for reduction
    tl.store(output_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    # Get array size from input tensors
    n_elements = a.shape[0]
    
    # Create output tensor for partial products
    partial_products = torch.zeros_like(a)
    
    # Launch kernel parameters
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s352_kernel[grid](
        a, b, partial_products, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute final dot product by summing all partial products
    dot = torch.sum(partial_products)
    
    return dot