import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    last_negative_index = -1
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        negative_mask = vals < 0.0
        
        # Find indices where condition is true
        condition_indices = tl.where(negative_mask & mask, current_offsets, -1)
        
        # Update last_negative_index for each true condition
        for i in range(BLOCK_SIZE):
            if i < BLOCK_SIZE and (block_start + i) < n_elements:
                idx = block_start + i
                if idx < n_elements:
                    val = tl.load(a_ptr + idx)
                    if val < 0.0:
                        last_negative_index = idx
    
    tl.store(result_ptr, last_negative_index)

def s331_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    grid = (1,)
    s331_kernel[grid](a, result, n_elements, BLOCK_SIZE)
    
    return result.item()