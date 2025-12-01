import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset array once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n_elements
    
    # Load indices with masking
    ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
    
    # Load a values with masking
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Load b values using indirect addressing with bounds checking
    b_mask = mask & (ip_vals >= 0) & (ip_vals < n_elements)
    b_vals = tl.load(b_ptr + ip_vals, mask=b_mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Only include valid products in sum
    valid_products = tl.where(mask, products, 0.0)
    
    # Store partial sum for this block
    partial_sum = tl.sum(valid_products)
    tl.store(output_ptr + pid, partial_sum)

def s4115_triton(a, b, ip):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s4115_kernel[(grid_size,)](
        a, b, ip, partial_sums, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial sums to get final result
    return partial_sums.sum()