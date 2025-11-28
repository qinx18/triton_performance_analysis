import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the last index where a[i] < 0
    # We need to process the entire array and find the maximum valid index
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Check which elements are negative
    negative_mask = a_vals < 0.0
    
    # Create valid indices for this block
    valid_indices = tl.where(mask & negative_mask, offsets, -1)
    
    # Find the maximum valid index in this block
    block_max = tl.max(valid_indices)
    
    # Store the result for this block
    tl.store(result_ptr + pid, block_max)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block results
    block_results = torch.full((num_blocks,), -1, dtype=torch.int32, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s331_kernel[grid](
        a, block_results, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find the maximum across all blocks to get the final result
    j = torch.max(block_results).item()
    
    # Convert to float as chksum
    chksum = float(j)
    
    return j