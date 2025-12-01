import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value in array a
    # Each block processes BLOCK_SIZE elements and finds local max
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements with mask, use -infinity for masked elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find maximum within this block
    block_max = tl.max(vals)
    
    # Store block maximum
    tl.store(output_ptr + tl.program_id(0), block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # Use block size of 1024
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block maxima
    block_maxs = torch.empty(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel to find block maxima
    s314_kernel[(grid_size,)](
        a, block_maxs, n_elements, BLOCK_SIZE
    )
    
    # If we have multiple blocks, recursively find max of block maxima
    while block_maxs.shape[0] > 1:
        n_blocks = block_maxs.shape[0]
        next_grid_size = triton.cdiv(n_blocks, BLOCK_SIZE)
        next_block_maxs = torch.empty(next_grid_size, dtype=a.dtype, device=a.device)
        
        s314_kernel[(next_grid_size,)](
            block_maxs, next_block_maxs, n_blocks, BLOCK_SIZE
        )
        
        block_maxs = next_block_maxs
    
    return block_maxs[0].item()