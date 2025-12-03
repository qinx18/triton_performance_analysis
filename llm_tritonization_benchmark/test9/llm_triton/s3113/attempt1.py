import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate offset range for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Load block of data with masking
    mask = indices < n_elements
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute absolute values
    abs_vals = tl.abs(a_vals)
    
    # Find maximum in this block
    block_max = tl.max(abs_vals, axis=0)
    
    # Store result
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a, abs):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary tensor to store block results
    block_results = torch.empty(num_blocks, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    s3113_kernel[(num_blocks,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Find maximum across all blocks
    max_val = torch.max(block_results)
    
    return max_val