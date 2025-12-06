import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the last index where a[i] < 0
    # We need to process the array in blocks and keep track of the maximum index found
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load values from array a
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=1.0)
    
    # Find indices where a[i] < 0
    condition = a_vals < 0.0
    
    # Get the actual indices
    indices = block_start + offsets
    
    # Set indices to -1 where condition is false, keep actual index where true
    valid_indices = tl.where(condition & mask, indices, -1)
    
    # Find the maximum valid index in this block
    max_idx = tl.max(valid_indices)
    
    # Store the result (each block stores its maximum index)
    tl.store(result_ptr + tl.program_id(0), max_idx)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor to store results from each block
    block_results = torch.full((num_blocks,), -1, dtype=torch.int32, device=a.device)
    
    # Launch kernel
    s331_kernel[(num_blocks,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Find the maximum index across all blocks (this gives us the last occurrence)
    j = torch.max(block_results).item()
    
    return j