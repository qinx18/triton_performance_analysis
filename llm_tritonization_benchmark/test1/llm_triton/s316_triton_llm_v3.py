import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the minimum value in the array using reduction
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with masking, use large value for masked elements
    data = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
    
    # Find minimum within this block
    block_min = tl.min(data, axis=0)
    
    # Store the block minimum
    tl.store(result_ptr + pid, block_min)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block results
    block_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel for first reduction
    grid = (num_blocks,)
    s316_kernel[grid](
        a, block_results, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # If we have multiple blocks, need to reduce further
    while block_results.shape[0] > 1:
        current_size = block_results.shape[0]
        new_num_blocks = triton.cdiv(current_size, BLOCK_SIZE)
        new_results = torch.empty(new_num_blocks, dtype=a.dtype, device=a.device)
        
        grid = (new_num_blocks,)
        s316_kernel[grid](
            block_results, new_results, current_size, BLOCK_SIZE=BLOCK_SIZE
        )
        
        block_results = new_results
    
    return block_results[0]