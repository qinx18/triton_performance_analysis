import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offset vectors
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Create mask for valid indices
    mask = idx < n
    
    # Load a values for this block
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Load ip values for this block (indices)
    ip_vals = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Load b values using indirect addressing with proper masking
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute dot product for this block
    products = a_vals * b_vals
    local_sum = tl.sum(products, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + pid, local_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    
    # Calculate grid size
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4115_kernel[grid](a, b, ip, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all partial results
    result = torch.sum(partial_sums)
    
    return result.item()