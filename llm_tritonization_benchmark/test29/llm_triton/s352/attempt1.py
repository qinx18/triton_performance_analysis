import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute dot product contribution for each element
    products = a_vals * b_vals
    
    # Store partial results for reduction
    tl.store(dot_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    n_elements = a.shape[0]
    device = a.device
    
    # Create output tensor for partial results
    partial_results = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s352_kernel[grid](
        a, b, partial_results,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_results)
    
    return dot