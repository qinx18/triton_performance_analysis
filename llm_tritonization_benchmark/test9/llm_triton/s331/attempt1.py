import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(
    a_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel finds the last index where a[i] < 0
    # We need to process the array in blocks and find the maximum valid index
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load array values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Find indices where a[i] < 0
    neg_mask = a_vals < 0.0
    valid_mask = mask & neg_mask
    
    # Get actual indices
    actual_indices = block_start + offsets
    
    # Set invalid indices to -1
    indices = tl.where(valid_mask, actual_indices, -1)
    
    # Find the maximum valid index in this block
    max_idx = tl.max(indices)
    
    # Store the result for this block
    tl.store(result_ptr + tl.program_id(0), max_idx)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary array to store results from each block
    block_results = torch.full((n_blocks,), -1, dtype=torch.int32, device=a.device)
    
    # Launch kernel
    grid = (n_blocks,)
    s331_kernel[grid](
        a,
        block_results,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Find the maximum index across all blocks
    j = torch.max(block_results).item()
    
    return j