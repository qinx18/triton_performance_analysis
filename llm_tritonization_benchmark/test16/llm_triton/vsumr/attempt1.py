import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate global indices for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid elements
    mask = indices < n_elements
    
    # Load data with masking
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute sum for this block
    block_sum = tl.sum(vals, axis=0)
    
    # Store the partial sum
    tl.store(result_ptr + tl.program_id(0), block_sum)

def vsumr_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (n_blocks,)
    vsumr_kernel[grid](
        a, partial_sums, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()