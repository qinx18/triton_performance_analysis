import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(
    a_ptr, b_ptr, ip_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load a values
    a_offsets = block_start + offsets
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Load indices
    ip_vals = tl.load(ip_ptr + a_offsets, mask=mask, other=0)
    
    # Gather b values using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store partial products for reduction
    tl.store(output_ptr + a_offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.numel()
    
    # Create output tensor for partial products
    partial_products = torch.zeros_like(a)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip, partial_products,
        n_elements,
        BLOCK_SIZE,
    )
    
    # Sum all partial products
    return partial_products.sum().item()