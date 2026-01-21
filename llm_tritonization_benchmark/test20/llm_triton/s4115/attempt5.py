import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load indices ip[i] - need to ensure they're valid indices
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Clamp ip_vals to valid range to avoid out-of-bounds access
    ip_vals = tl.where(ip_vals < 0, 0, ip_vals)
    
    # Load b[ip[i]] values using gathered indices
    # Use mask to only load valid elements
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products only for valid indices
    products = tl.where(mask, a_vals * b_vals, 0.0)
    
    # Sum within this block
    block_sum = tl.sum(products)
    
    # Store block sum to output array
    tl.store(output_ptr + pid, block_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for block sums
    block_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4115_kernel[grid](
        a, b, ip, block_sums,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all block results
    return torch.sum(block_sums).item()