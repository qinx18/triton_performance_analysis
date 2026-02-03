import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize result with first element
    if tl.program_id(0) == 0:
        first_val = tl.load(a_ptr)
        tl.store(result_ptr, first_val)
    
    # Process blocks starting from index 1
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from current block
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in current block
        block_min = tl.min(vals)
        
        # Update global minimum atomically
        if tl.program_id(0) == 0:
            current_min = tl.load(result_ptr)
            if block_min < current_min:
                tl.store(result_ptr, block_min)

def s316_triton(a):
    N = a.shape[0]
    
    # Create result tensor
    result = torch.full((1,), float('inf'), dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s316_kernel[grid](a, result, N, BLOCK_SIZE)
    
    return result.item()