import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program id
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load elements from a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Apply condition: if a[i] > 0, include in sum, else 0
    condition = a_vals > 0.0
    conditional_vals = tl.where(condition, a_vals, 0.0)
    
    # Sum the conditional values in this block
    block_sum = tl.sum(conditional_vals)
    
    # Store the block sum
    tl.store(output_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s3111_kernel[grid](
        a, partial_sums, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()