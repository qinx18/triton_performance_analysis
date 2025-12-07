import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values and compute absolute values
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find maximum in this block
    local_max = tl.max(abs_vals)
    
    # Store block result
    tl.store(result_ptr + pid, local_max)

def s3113_triton(a, abs):
    n_elements = a.numel()
    
    # Calculate number of blocks needed
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary result tensor for block maximums
    block_results = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3113_kernel[(n_blocks,)](
        a, block_results, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find final maximum across all blocks
    final_max = torch.max(block_results).item()
    
    return final_max