import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
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
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products and sum
    products = a_vals * b_vals
    block_sum = tl.sum(products)
    
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