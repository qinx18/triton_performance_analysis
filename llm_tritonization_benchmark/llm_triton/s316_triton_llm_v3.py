import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction to find minimum value
    # Each block will find local minimum, then we'll reduce across blocks
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with masking
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
    
    # Find minimum within this block
    local_min = tl.min(vals)
    
    # Store local minimum for this block
    tl.store(output_ptr + tl.program_id(0), local_min)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # First pass: reduce within blocks
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Temporary storage for block results
    block_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel for first reduction
    grid = (num_blocks,)
    s316_kernel[grid](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU for simplicity (small number of blocks)
    final_min = torch.min(block_results)
    
    return final_min