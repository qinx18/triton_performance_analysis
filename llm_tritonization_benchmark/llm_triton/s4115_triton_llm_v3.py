import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(
    a_ptr, b_ptr, ip_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start and end
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load indices for indirect addressing
    ip_vals = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Sum the products in this block
    block_sum = tl.sum(products)
    
    # Store the block sum
    tl.store(output_ptr + pid, block_sum)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    
    # Calculate number of blocks needed
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s4115_kernel[(n_blocks,)](
        a, b, ip, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum all partial sums to get final result
    total_sum = torch.sum(partial_sums)
    
    return total_sum