import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize global max and index
    global_max = float('-inf')
    global_index = 0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max within this block using tl.max and tl.argmax
        block_max = tl.max(vals, axis=0)
        
        # Check if this block max is greater than global max
        if block_max > global_max:
            global_max = block_max
            # Find the index within the block
            block_argmax = tl.argmax(vals, axis=0)
            global_index = block_start + block_argmax
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, global_max)
        tl.store(result_ptr + 1, global_index.to(tl.float32))

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