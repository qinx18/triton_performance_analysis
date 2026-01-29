import triton
import triton.language as tl
import torch

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    j = -1
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=1.0)
        
        negative_mask = a_vals < 0.0
        valid_mask = mask & negative_mask
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            indices = tl.where(valid_mask, current_offsets, -1)
            max_idx = tl.max(indices)
            if max_idx >= 0:
                j = max_idx
    
    tl.store(result_ptr, j)

def s331_triton(a):
    N = a.shape[0]
    
    result = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s331_kernel[grid](
        a, result, N, BLOCK_SIZE
    )
    
    return result.item()