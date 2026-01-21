import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each program handles one block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load a values
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load ip values (indices for b array)
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Load b values using indirect addressing
    # Apply mask to ensure we only load valid b values
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute partial products
    products = a_vals * b_vals
    
    # Sum the products for this block
    block_sum = tl.sum(products, axis=0)
    
    # Store the block sum
    tl.store(result_ptr + pid, block_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4115_kernel[grid](
        a, b, ip, partial_sums,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final sum
    total_sum = torch.sum(partial_sums)
    
    return total_sum