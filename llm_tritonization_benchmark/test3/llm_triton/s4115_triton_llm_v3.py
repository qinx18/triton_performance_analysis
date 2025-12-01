import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices and a values
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Gather b values using indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store products for reduction
    tl.store(output_ptr + offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create temporary buffer for partial products
    temp_output = torch.zeros_like(a)
    
    # Launch kernel
    s4115_kernel[grid](
        a, b, ip, temp_output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final result
    return temp_output.sum().item()