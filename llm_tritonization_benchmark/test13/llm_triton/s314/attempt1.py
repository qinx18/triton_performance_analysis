import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel to find maximum value
    # Each block processes BLOCK_SIZE elements and finds local max
    # Then we need to reduce across blocks
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load block of data
    vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=float('-inf'))
    
    # Find maximum in this block
    block_max = tl.max(vals)
    
    # Store block maximum
    tl.store(result_ptr + tl.program_id(0), block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary tensor to store block maxima
    block_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to find maximum in each block
    grid = (num_blocks,)
    s314_kernel[grid](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Find global maximum from block results
    # For small number of blocks, use CPU reduction
    if num_blocks <= 1024:
        result = torch.max(block_results).item()
    else:
        # For large number of blocks, recursively apply reduction
        while block_results.shape[0] > 1:
            n_elements = block_results.shape[0]
            num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
            if num_blocks == 1:
                result = torch.max(block_results).item()
                break
            
            new_block_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
            grid = (num_blocks,)
            s314_kernel[grid](
                block_results, new_block_results, n_elements, BLOCK_SIZE
            )
            block_results = new_block_results
        
        if block_results.shape[0] == 1:
            result = block_results[0].item()
        else:
            result = torch.max(block_results).item()
    
    return result