import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Load indices
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask, other=0)
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store products
    tl.store(output_ptr + block_start + offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for partial products
    output = torch.zeros_like(a)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final result
    result = torch.sum(output)
    return result