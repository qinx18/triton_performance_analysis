import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices and a values
    ip_offsets = block_start + offsets
    indices = tl.load(ip_ptr + ip_offsets, mask=mask, other=0)
    a_vals = tl.load(a_ptr + ip_offsets, mask=mask, other=0.0)
    
    # Create mask for valid b indices
    b_mask = mask & (indices >= 0) & (indices < n_elements)
    
    # Load b values using indirect addressing
    safe_indices = tl.where(b_mask, indices, 0)
    b_vals = tl.load(b_ptr + safe_indices)
    
    # Compute products only for valid entries
    products = tl.where(b_mask, a_vals * b_vals, 0.0)
    
    # Store results
    tl.store(output_ptr + ip_offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor
    output = torch.zeros_like(a)
    
    # Launch kernel
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s4115_kernel[grid](
        a, b, ip, output, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return sum of all products
    return torch.sum(output).item()