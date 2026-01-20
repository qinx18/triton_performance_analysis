import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < N
    
    # Load data with masking
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask, other=0)
    
    # Gather from b using indirect addressing
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Sum the products for this block
    block_sum = tl.sum(products)
    
    # Store the partial sum
    tl.store(output_ptr + pid, block_sum)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (grid_size,)
    s4115_kernel[grid](a, b, ip, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all partial results and return as scalar tensor
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()