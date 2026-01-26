import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Global min across all blocks
    global_min = float('inf')
    
    # Process blocks sequentially to maintain reduction semantics
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(vals, axis=0)
        
        # Update global minimum
        global_min = tl.minimum(global_min, block_min)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, global_min)

def s316_triton(a):
    N = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    # Launch single program to handle entire reduction
    s316_kernel[(1,)](
        a, result, N, BLOCK_SIZE
    )
    
    return result.item()