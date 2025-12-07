import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Load indices
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask, other=0)
    
    # Load b values using gathered indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store products for reduction
    tl.store(output_ptr + block_start + offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for products
    products = torch.zeros_like(a)
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s4115_kernel[grid](
        a, b, ip, products,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products
    return products.sum().item()