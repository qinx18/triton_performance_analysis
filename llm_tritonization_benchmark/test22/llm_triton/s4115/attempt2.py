import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < n
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Create mask for valid b indices to avoid out of bounds access
    b_mask = mask
    
    # Gather from b using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=b_mask, other=0.0)
    
    # Compute products only for valid elements
    products = tl.where(mask, a_vals * b_vals, 0.0)
    
    # Sum the products
    block_sum = tl.sum(products)
    
    # Store the partial sum
    tl.store(output_ptr + pid, block_sum)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4115_kernel[grid](
        a, b, ip, partial_sums,
        N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial sums to get final result
    total_sum = torch.sum(partial_sums)
    
    return total_sum