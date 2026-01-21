import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and create offset vector
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product contribution
    products = a_vals * b_vals
    
    # Store products for later reduction
    tl.store(result_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    N = a.shape[0]
    
    # Create output tensor for intermediate products
    products = torch.zeros_like(a)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s352_kernel[grid](
        a, b, products, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final dot product
    dot = torch.sum(products)
    
    return dot