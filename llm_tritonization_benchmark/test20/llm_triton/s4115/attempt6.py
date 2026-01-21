import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles one block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices first
    ip_offsets = block_start + offsets
    indices = tl.load(ip_ptr + ip_offsets, mask=mask, other=0)
    
    # Load a values
    a_vals = tl.load(a_ptr + ip_offsets, mask=mask, other=0.0)
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Store partial results
    tl.store(output_ptr + ip_offsets, products, mask=mask)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for partial results
    partial_results = torch.zeros_like(a)
    
    # Launch kernel
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s4115_kernel[grid](
        a, b, ip, partial_results, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final sum
    return torch.sum(partial_results).item()