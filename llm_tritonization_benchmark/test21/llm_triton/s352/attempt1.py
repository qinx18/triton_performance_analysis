import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the current block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < N
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product contribution
    products = a_vals * b_vals
    
    # Store partial results
    tl.store(result_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for partial products
    partial_products = torch.zeros_like(a)
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s352_kernel[grid](
        a, b, partial_products, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial products to get final dot product
    dot = torch.sum(partial_products)
    
    return dot