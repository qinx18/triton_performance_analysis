import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    mask = block_offsets < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + block_offsets, mask=mask, other=0.0)
    
    # Load indices, ensuring they are within bounds
    indices = tl.load(ip_ptr + block_offsets, mask=mask, other=0)
    
    # Create a mask for valid indices (within b array bounds)
    b_mask = mask & (indices >= 0) & (indices < n_elements)
    
    # Load b values using indices, with proper masking
    b_vals = tl.load(b_ptr + indices, mask=b_mask, other=0.0)
    
    # Compute products, only for valid entries
    products = tl.where(b_mask, a_vals * b_vals, 0.0)
    
    # Store products
    tl.store(output_ptr + block_offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output buffer for products
    output = torch.zeros_like(a)
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s4115_kernel[grid](
        a, b, ip, output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return sum of all products
    return output.sum().item()