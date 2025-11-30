import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load indices and gather from b
    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store to output for later reduction
    tl.store(output_ptr + offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    
    # Create output tensor for partial products
    output = torch.zeros_like(a)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reduce to get final sum
    return torch.sum(output).item()