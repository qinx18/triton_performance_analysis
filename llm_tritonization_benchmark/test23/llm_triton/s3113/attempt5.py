import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute absolute values and find block maximum
    abs_vals = tl.abs(vals)
    block_max = tl.max(abs_vals, axis=0)
    
    # Store block result
    tl.store(result_ptr + block_id, block_max)

def s3113_triton(a):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create temporary array to store block maxima
    block_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    
    s3113_kernel[grid](
        a, block_results, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find the maximum across all blocks
    final_max = torch.max(block_results).item()
    
    # Initialize with abs(a[0]) like the C code
    a0_abs = torch.abs(a[0]).item()
    final_max = max(final_max, a0_abs)
    
    return final_max