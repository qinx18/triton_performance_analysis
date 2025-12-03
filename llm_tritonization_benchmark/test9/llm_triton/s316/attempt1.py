import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel to find minimum value
    # Each block processes BLOCK_SIZE elements and finds local minimum
    # Then we need a final reduction across blocks
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Load data with masking
    mask = current_offsets < n_elements
    # Use large value for masked elements so they don't affect minimum
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
    
    # Find minimum within this block
    block_min = tl.min(a_vals)
    
    # Store block minimum
    tl.store(result_ptr + block_id, block_min)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # First reduction: each block finds its minimum
    block_results = torch.empty(n_blocks, dtype=a.dtype, device=a.device)
    
    s316_kernel[(n_blocks,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU for small number of blocks
    if n_blocks == 1:
        return block_results[0]
    else:
        return torch.min(block_results)