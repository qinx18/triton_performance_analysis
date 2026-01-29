import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each block processes BLOCK_SIZE elements in parallel
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise product
    products = a_vals * b_vals
    
    # Store products for later reduction
    tl.store(result_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    N = a.shape[0]
    
    # Create temporary array for products
    products = torch.zeros_like(a)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s352_kernel[grid](
        a, b, products, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reduce to get dot product
    dot = torch.sum(products)
    
    return dot