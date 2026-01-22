import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Apply condition: if a[i] > 0, include in sum, else 0
    condition_mask = a_vals > 0.0
    filtered_vals = tl.where(condition_mask, a_vals, 0.0)
    block_sum = tl.sum(filtered_vals)
    
    # Store block sum
    tl.store(output_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output for block sums
    block_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (n_blocks,)
    s3111_kernel[grid](a, block_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return total sum as scalar
    return torch.sum(block_sums).item()