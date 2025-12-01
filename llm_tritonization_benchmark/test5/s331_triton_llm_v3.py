import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    last_negative_idx = -1
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        negative_mask = a_vals < 0.0
        
        # Find indices where condition is true
        valid_indices = tl.where(negative_mask & mask, current_offsets, -1)
        
        # Update last_negative_idx for each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                if valid_indices[i] >= 0:
                    last_negative_idx = valid_indices[i]
    
    # Store the result (only thread 0 writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, last_negative_idx)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    grid = (1,)
    s331_kernel[grid](
        a, result, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()