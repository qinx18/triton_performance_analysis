import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Load indices from ip
    ip_vals = tl.load(ip_ptr + block_start + offsets, mask=mask, other=0)
    
    # Ensure indices are within bounds for b array access
    valid_indices = (ip_vals >= 0) & (ip_vals < n_elements) & mask
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=valid_indices, other=0.0)
    
    # Compute products, only for valid entries
    products = tl.where(valid_indices, a_vals * b_vals, 0.0)
    
    # Store partial results
    tl.store(output_ptr + block_start + offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for partial results
    output = torch.zeros_like(a)
    
    # Launch kernel
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s4115_kernel[grid](
        a, b, ip, output, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    return output.sum().item()