import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid elements
    mask = indices < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load indices for b
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Load b values using gathered indices
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store products for later reduction
    tl.store(output_ptr + indices, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    
    # Create output tensor for partial products
    partial_products = torch.zeros_like(a)
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s4115_kernel[grid](
        a, b, ip, partial_products,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final result
    result = torch.sum(partial_products)
    
    return result