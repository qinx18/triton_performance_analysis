import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Global reduction using multiple blocks
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Load data with masking
    mask = current_offsets < n_elements
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Sum within block
    block_sum = tl.sum(a_vals)
    
    # Store partial sum
    tl.store(result_ptr + block_id, block_sum)

def s311_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s311_kernel[grid](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    total_sum = torch.sum(partial_sums)
    
    return total_sum