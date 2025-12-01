import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value in the array
    # Each block processes BLOCK_SIZE elements and finds local max
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values with mask, use -inf for masked elements
    values = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find maximum within this block
    block_max = tl.max(values, axis=0)
    
    # Store the block maximum
    tl.store(result_ptr + tl.program_id(0), block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary array to store block maxima
    block_maxs = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to find block maxima
    grid = (num_blocks,)
    s314_kernel[grid](
        a, block_maxs, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find the global maximum from block maxima
    x = torch.max(block_maxs).item()
    
    return x