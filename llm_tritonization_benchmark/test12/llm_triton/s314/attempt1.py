import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find maximum element using reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    block_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    # Process all blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.maximum(block_max, vals)
    
    # Reduce within block
    result = tl.max(block_max, axis=0)
    
    # Store result (single work item stores)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(a_ptr + n_elements, result)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Extend tensor by 1 to store result
    extended_a = torch.cat([a, torch.zeros(1, dtype=a.dtype, device=a.device)])
    
    # Launch single program
    grid = (1,)
    s314_kernel[grid](
        extended_a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return extended_a[n_elements].item()