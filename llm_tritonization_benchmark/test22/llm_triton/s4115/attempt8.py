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
    
    # Load indices and ensure they are valid
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask, other=0)
    
    # Load b values using indices - need to mask for valid indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = tl.where(mask, a_vals * b_vals, 0.0)
    
    # Sum within block
    block_sum = tl.sum(products)
    
    # Store result (each block stores its partial sum)
    tl.store(output_ptr + tl.program_id(0), block_sum)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros((grid_size,), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s4115_kernel[(grid_size,)](
        a, b, ip, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Sum all partial results
    return partial_sums.sum().item()