import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Global reduction across all blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block-local max and index
    block_max = float('-inf')
    block_index = 0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max within this block
        for i in range(BLOCK_SIZE):
            if current_offsets[i] < n_elements:
                if vals[i] > block_max:
                    block_max = vals[i]
                    block_index = current_offsets[i]
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, block_max)
        tl.store(result_ptr + 1, block_index.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create result tensor [max_value, max_index]
    result = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single block for global reduction
    grid = (1,)
    s315_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    max_val = result[0].item()
    max_idx = int(result[1].item())
    
    return max_val, max_idx