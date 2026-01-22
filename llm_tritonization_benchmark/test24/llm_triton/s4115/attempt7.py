import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices - need to clamp to valid range
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask, other=0)
    
    # Load a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Load b values using indirect addressing - clamp indices to valid range
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store products for reduction
    tl.store(output_ptr + block_start + offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for partial products
    output = torch.zeros_like(a)
    
    # Ensure ip indices are within bounds
    ip_clamped = torch.clamp(ip, 0, b.shape[0] - 1)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4115_kernel[grid](
        a, b, ip_clamped, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products
    result = torch.sum(output)
    return result